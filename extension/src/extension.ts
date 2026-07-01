import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { exec, spawn } from 'child_process';

// ── config (bring your own model) ──────────────────────────────────
function cfg<T>(k: string, d: T): T {
  return vscode.workspace.getConfiguration('apprentice').get<T>(k) ?? d;
}
type Msg = { role: string; content: string };

// ── LLM backend: any OpenAI-compatible endpoint, streaming ─────────
async function llmStream(messages: Msg[], onChunk: (s: string) => void): Promise<string> {
  const url = cfg('apiUrl', 'https://api.openai.com/v1/chat/completions');
  const key = cfg('apiKey', '');
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (key) { headers['Authorization'] = `Bearer ${key}`; }
  const res = await fetch(url, {
    method: 'POST', headers,
    body: JSON.stringify({ model: cfg('model', 'gpt-4o-mini'), messages, temperature: 0.2, stream: true }),
  });
  if (!res.ok || !res.body) { throw new Error(`LLM HTTP ${res.status}`); }
  const reader = (res.body as any).getReader();
  const dec = new TextDecoder();
  let buf = '', full = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) { break; }
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
      if (!line.startsWith('data:')) { continue; }
      const data = line.slice(5).trim();
      if (data === '[DONE]') { return full; }
      try { const c = JSON.parse(data).choices[0].delta.content; if (c) { full += c; onChunk(c); } } catch { /**/ }
    }
  }
  return full;
}

// ── Claude CLI backend (Claude account as brain — no API key) ──────
// Runs `claude -p` as a clean chat function: tools disabled, user memory skipped,
// neutral cwd, and the whole prompt (incl. system instruction) via stdin so it
// behaves as a stateless responder instead of a full agent.
const NO_TOOLS = 'Bash Edit Write Read Glob Grep WebSearch WebFetch Task TodoWrite NotebookEdit';
function claudeCli(messages: Msg[]): Promise<string> {
  let sys = ''; const parts: string[] = [];
  for (const m of messages) {
    if (m.role === 'system') { sys += (sys ? '\n\n' : '') + m.content; }
    else if (m.role === 'user') { parts.push(m.content); }
    else { parts.push(`[assistant]\n${m.content}`); }
  }
  const stdin = (sys ? `SYSTEM:\n${sys}\n\n` : '') + parts.join('\n\n');
  const args = ['-p', '--setting-sources', '', '--disallowedTools', NO_TOOLS];
  return new Promise((resolve, reject) => {
    const p = spawn('claude', args, { cwd: os.tmpdir(), shell: process.platform === 'win32' });
    let out = '', err = '';
    p.stdout.on('data', (d) => { out += d.toString(); });
    p.stderr.on('data', (d) => { err += d.toString(); });
    p.on('error', reject);
    p.on('close', () => { const t = out.trim(); if (t) { resolve(t); } else { reject(new Error(err.slice(0, 300) || 'claude-cli: empty output')); } });
    p.stdin.write(stdin); p.stdin.end();
  });
}

// dispatch: claude-cli (non-streaming) or OpenAI-compatible HTTP (streaming)
async function llmCall(messages: Msg[], onChunk: (s: string) => void): Promise<string> {
  if (cfg<string>('backend', 'openai') === 'claude-cli') {
    const r = await claudeCli(messages); onChunk(r); return r;
  }
  return llmStream(messages, onChunk);
}

// ── workspace context ──────────────────────────────────────────────
let lastEditor: vscode.TextEditor | undefined;
function root(): string { return vscode.workspace.workspaceFolders?.[0].uri.fsPath || process.cwd(); }
function abs(p: string): string { return path.isAbsolute(p) ? p : path.join(root(), p); }
function activeNote(): string {
  const ed = lastEditor || vscode.window.activeTextEditor;
  if (!ed) { return ''; }
  const name = vscode.workspace.asRelativePath(ed.document.uri);
  if (!ed.selection.isEmpty) { return `[selection — ${name}]\n${ed.document.getText(ed.selection)}`; }
  let t = ed.document.getText();
  if (t.length > 8000) { t = t.slice(0, 8000) + '\n…(truncated)'; }
  return `[open file — ${name}]\n${t}`;
}

// ── agent system prompt (lessons baked in: investigate, never bluff) ─
const SYSTEM = `You are 'apprentice', an agentic coding assistant inside VS Code. Reply in the user's language.
Greetings/chitchat/opinions: answer immediately with finish, no tools.
Concrete tasks (read/search/edit/run) or "what/where is X" questions about the codebase: use tools to find the real answer.
Output exactly ONE JSON object per turn (no markdown, no prose):
{"thought":"<short>","action":"<tool>","args":{...}}

Tools:
- tree        {"path":"."}                         project structure overview
- list_dir    {"path":"."}                         list a folder
- read_file   {"path":"rel"}                        read a file
- search_code {"query":"text","path":"."}           grep across files (exact text)
- code_search {"query":"meaning"}                   semantic code search (finds relevant code by meaning; if configured)
- edit_file   {"path":"...","old":"...","new":"..."} targeted replace (diff-approved)
- write_file  {"path":"...","content":"..."}        create/overwrite (approved)
- run_command {"command":"..."}                     shell (approved)
- finish      {"message":"<final answer>"}          done

Rules: one JSON only. Confirm with read_file/search_code before editing. Never invent facts about the
codebase — if you don't know a filename/path/value, FIND it with the tools; do not paper over it with
generic statements like "it depends on the structure". Only say "couldn't find it" after actually looking.`;

// ── tools ──────────────────────────────────────────────────────────
const SKIP = new Set(['.git', '__pycache__', 'node_modules', '.venv', 'out', 'dist', '.next', 'build']);
const EXTS = ['.py', '.js', '.ts', '.tsx', '.jsx', '.md', '.json', '.yml', '.yaml', '.html', '.css', '.java', '.c', '.cpp', '.h', '.rs', '.go', '.sh'];
const TREE_IGNORE = new Set([...SKIP, 'data', 'target', '.idea', '.vscode']);

function buildTree(dir: string, prefix: string, depth: number, lines: string[]): void {
  if (depth > 2 || lines.length > 160) { return; }
  let entries: fs.Dirent[];
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
  entries = entries.filter((e) => !TREE_IGNORE.has(e.name) && !(e.name.startsWith('.') && e.name !== '.gitignore')).slice(0, 50);
  for (const e of entries) {
    lines.push(prefix + (e.isDirectory() ? '[D] ' : '') + e.name);
    if (e.isDirectory()) { buildTree(path.join(dir, e.name), prefix + '  ', depth + 1, lines); }
  }
}
function grep(dir: string, q: string, hits: string[]): void {
  if (hits.length >= 40) { return; }
  let entries: fs.Dirent[];
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
  for (const e of entries) {
    if (hits.length >= 40) { return; }
    const fp = path.join(dir, e.name);
    if (e.isDirectory()) { if (!SKIP.has(e.name)) { grep(fp, q, hits); } }
    else if (EXTS.includes(path.extname(e.name))) {
      try {
        fs.readFileSync(fp, 'utf8').split('\n').forEach((ln, i) => {
          if (hits.length < 40 && ln.includes(q)) { hits.push(`${vscode.workspace.asRelativePath(fp)}:${i + 1}: ${ln.trim().slice(0, 150)}`); }
        });
      } catch { /**/ }
    }
  }
}
function runCmd(cmd: string): Promise<string> {
  return new Promise((res) => exec(cmd, { cwd: root(), timeout: 60000, maxBuffer: 1 << 20 },
    (err, so, se) => res((`${so}\n${se}`).trim() || (err ? String(err) : '(no output)'))));
}
function approve(desc: string): Thenable<boolean> {
  return vscode.window.showWarningMessage(desc, { modal: true }, 'Apply').then((r) => r === 'Apply');
}

async function runTool(action: string, args: any): Promise<string> {
  try {
    if (action === 'tree') { const l: string[] = []; buildTree(abs(args.path || '.'), '', 0, l); return l.join('\n').slice(0, 3000) || '(empty)'; }
    if (action === 'list_dir') {
      return fs.readdirSync(abs(args.path || '.'), { withFileTypes: true })
        .map((e) => (e.isDirectory() ? '[D] ' : '') + e.name).join('\n') || '(empty)';
    }
    if (action === 'read_file') {
      const t = fs.readFileSync(abs(args.path), 'utf8');
      return t.length > 6000 ? t.slice(0, 6000) + '\n…(truncated — use search_code for a specific part)' : t;
    }
    if (action === 'search_code') { const h: string[] = []; grep(abs(args.path || '.'), args.query || '', h); return h.length ? h.join('\n') : 'No matches.'; }
    if (action === 'code_search') {
      const u = cfg('codeRagUrl', '');
      if (!u) { return 'code_search disabled (set apprentice.codeRagUrl).'; }
      const r = await fetch(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: args.query || '', k: 6 }) });
      if (!r.ok) { return `code_search HTTP ${r.status}`; }
      return ((await r.json() as any).results || '(no relevant code)').slice(0, 4000);
    }
    if (action === 'edit_file') {
      const fp = abs(args.path); const c = fs.readFileSync(fp, 'utf8');
      const n = c.split(args.old).length - 1;
      if (n === 0) { return 'old not found — read_file first.'; }
      if (n > 1) { return `old matches ${n} places — make it more specific.`; }
      if (!(await approve(`apprentice will edit ${args.path}. Apply?`))) { return 'User declined.'; }
      fs.writeFileSync(fp, c.replace(args.old, args.new)); return `Edited ${args.path}.`;
    }
    if (action === 'write_file') {
      if (!(await approve(`apprentice will write ${args.path}. Apply?`))) { return 'User declined.'; }
      fs.mkdirSync(path.dirname(abs(args.path)), { recursive: true });
      fs.writeFileSync(abs(args.path), args.content || ''); return `Wrote ${args.path}.`;
    }
    if (action === 'run_command') {
      if (!(await approve(`apprentice will run:\n${args.command}\nApply?`))) { return 'User declined.'; }
      return await runCmd(args.command);
    }
    return `Unknown tool: ${action}`;
  } catch (e: any) { return `Error: ${e.message}`; }
}

function parseAction(text: string): any {
  let t = text.trim().replace(/^```(json)?/i, '').replace(/```$/, '').trim();
  try { return JSON.parse(t); } catch { /**/ }
  let depth = 0, start = -1;
  for (let i = 0; i < t.length; i++) {
    if (t[i] === '{') { if (depth === 0) { start = i; } depth++; }
    else if (t[i] === '}') { if (--depth === 0 && start >= 0) { try { return JSON.parse(t.slice(start, i + 1)); } catch { /**/ } } }
  }
  return null;
}

// ── chat session (agent loop) ──────────────────────────────────────
const MAX_STEPS = 24;
const VAGUE = /(it depends on|depends on the (structure|setup)|구조에 따라|내부 구조에 따라|에 따라 달라)/i;

class ChatSession {
  history: Msg[] = [];
  lastQuestion = '';
  lastAnswer = '';
  constructor(public webview: vscode.Webview) {
    webview.options = { enableScripts: true };
    webview.html = html();
    webview.onDidReceiveMessage((m) => {
      if (m.type === 'ask') { this.run(m.text); }
      else if (m.type === 'correct') { this.saveCorrection(m.correction); }
    });
  }
  post(m: any) { this.webview.postMessage(m); }

  // Capture a correction for the self-improvement loop. The panel can't run the embedder, so it
  // only records the raw correction; `apprentice digest` later distills it into a lesson.
  saveCorrection(correction: string) {
    try {
      const dir = path.join(root(), '.apprentice');
      fs.mkdirSync(dir, { recursive: true });
      const rec = { ts: Math.floor(Date.now() / 1000), task: this.lastQuestion, answer: this.lastAnswer, correction };
      fs.appendFileSync(path.join(dir, 'corrections.jsonl'), JSON.stringify(rec) + '\n', 'utf8');
      this.post({ type: 'corrected' });
    } catch (e: any) { this.post({ type: 'tool', text: `[correction failed] ${e.message}` }); }
  }

  // Self-improvement at query time: pull relevant lessons from the local service and inject them
  // as an authoritative [project knowledge] block (the injection that measured 0/4 -> 4/4 accuracy).
  async fetchLessons(query: string): Promise<string> {
    const url = cfg('lessonsUrl', '');
    if (!url) { return ''; }
    try {
      const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, k: 3 }) });
      if (!r.ok) { return ''; }
      return ((await r.json()) as any).knowledge || '';
    } catch { return ''; }
  }

  async run(text: string) {
    this.post({ type: 'user', text });
    this.lastQuestion = text;
    const ctx = activeNote();
    const know = await this.fetchLessons(text);
    if (know) { this.post({ type: 'tool', text: '[applied project lessons]' }); }
    const first = (know ? `${know}\n\n` : '') + (ctx ? `${ctx}\n\n` : '') + `[task]\n${text}\n(workspace: ${root()})`;
    const messages: Msg[] = [{ role: 'system', content: SYSTEM }, ...this.history.slice(-12), { role: 'user', content: first }];
    let investigated = 0, nudged = false;
    for (let step = 0; step < MAX_STEPS; step++) {
      if (messages.length > 16) { messages.splice(1, messages.length - 13); }
      this.post({ type: 'think-start' });
      let raw = '';
      try { raw = await llmCall(messages, (c) => this.post({ type: 'chunk', text: c })); }
      catch (e: any) { this.post({ type: 'tool', text: `[error] ${e.message}` }); return; }
      this.post({ type: 'think-end' });
      const act = parseAction(raw);
      if (!act) { messages.push({ role: 'assistant', content: raw }, { role: 'user', content: 'Output exactly one JSON object.' }); continue; }
      if (act.action === 'finish') {
        const msg = act.args?.message || 'done';
        if (!nudged && investigated < 3 && VAGUE.test(msg)) {
          nudged = true;
          messages.push({ role: 'assistant', content: JSON.stringify(act) },
            { role: 'user', content: 'That is too generic. Do not guess — use the tools to find the real filename/path/value and answer concretely.' });
          continue;
        }
        this.lastAnswer = msg;
        this.post({ type: 'final', text: msg });
        this.history.push({ role: 'user', content: text }, { role: 'assistant', content: msg });
        return;
      }
      this.post({ type: 'act', text: `${act.action} ${JSON.stringify(act.args || {}).slice(0, 100)}` });
      const obs = await runTool(act.action, act.args || {});
      if (['read_file', 'list_dir', 'search_code', 'code_search', 'tree'].includes(act.action)) { investigated++; }
      this.post({ type: 'tool', text: obs.length > 400 ? obs.slice(0, 400) + ' …' : obs });
      messages.push({ role: 'assistant', content: JSON.stringify(act) }, { role: 'user', content: `observation:\n${obs.slice(0, 2000)}` });
    }
    this.post({ type: 'final', text: '(step limit reached)' });
  }
}

export function activate(ctx: vscode.ExtensionContext) {
  lastEditor = vscode.window.activeTextEditor;
  ctx.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor((e) => { if (e) { lastEditor = e; } }),
    vscode.window.registerWebviewViewProvider('apprentice.chat', {
      resolveWebviewView: (v) => { new ChatSession(v.webview); },
    }),
    vscode.commands.registerCommand('apprentice.openPanel', () => {
      const p = vscode.window.createWebviewPanel('apprenticePanel', 'apprentice', vscode.ViewColumn.Beside, { enableScripts: true, retainContextWhenHidden: true });
      new ChatSession(p.webview);
    }),
  );
}
export function deactivate() { /**/ }

function html(): string {
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  body{font-family:var(--vscode-font-family);font-size:13px;color:var(--vscode-foreground);margin:0;padding:8px;display:flex;flex-direction:column;height:100vh;box-sizing:border-box}
  #log{flex:1;overflow:auto}
  .u{color:var(--vscode-textLink-foreground);font-weight:600;margin:10px 0 2px}
  .think{white-space:pre-wrap;color:var(--vscode-descriptionForeground);font-size:12px}
  .act{color:var(--vscode-charts-blue);font-size:12px;margin:4px 0}
  .tool{color:var(--vscode-descriptionForeground);font-size:11px;white-space:pre-wrap;border-left:2px solid var(--vscode-panel-border);padding-left:6px;margin:2px 0 8px}
  .final{white-space:pre-wrap;margin:6px 0 2px;line-height:1.5}
  .teach{margin:0 0 12px}
  .tlink{color:var(--vscode-descriptionForeground);font-size:11px;text-decoration:none;cursor:pointer}
  .tbox{display:none;margin-top:4px}
  .tinput{width:100%;box-sizing:border-box;background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border);border-radius:6px;padding:5px;font-family:inherit;resize:none}
  .tnote{color:var(--vscode-charts-green);font-size:11px}
  #bar{display:flex;gap:6px;margin-top:6px}
  #q{flex:1;background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border);border-radius:6px;padding:6px;resize:none;font-family:inherit}
  button{background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0;border-radius:6px;padding:6px 12px;cursor:pointer}
  </style></head><body>
  <div id="log"><div class="tool">apprentice — reads your files, searches your codebase, edits with approval. Bring your own model.</div></div>
  <div id="bar"><textarea id="q" rows="2" placeholder="Ask… (Enter)"></textarea><button id="send">Send</button></div>
  <script>
  const vs=acquireVsCodeApi(),log=document.getElementById('log'),q=document.getElementById('q');let cur=null;
  function add(c,t){const d=document.createElement('div');d.className=c;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
  function send(){const t=q.value.trim();if(!t)return;q.value='';vs.postMessage({type:'ask',text:t});}
  document.getElementById('send').onclick=send;
  q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
  let lastTeach=null;
  function teach(){
    const w=document.createElement('div');w.className='teach';
    const link=document.createElement('a');link.className='tlink';link.textContent='👎 wrong? teach it';
    const box=document.createElement('div');box.className='tbox';
    const ta=document.createElement('textarea');ta.className='tinput';ta.rows=2;ta.placeholder='the correct answer / what it should remember next time';
    const send=document.createElement('button');send.textContent='Teach';
    box.appendChild(ta);box.appendChild(send);w.appendChild(link);w.appendChild(box);
    link.onclick=()=>{box.style.display='block';ta.focus();};
    send.onclick=()=>{const c=ta.value.trim();if(!c)return;vs.postMessage({type:'correct',correction:c});box.innerHTML='<span class=tnote>saved — run \`apprentice digest\` to turn it into a lesson</span>';};
    log.appendChild(w);log.scrollTop=log.scrollHeight;return w;
  }
  window.addEventListener('message',e=>{const m=e.data;
    if(m.type==='user')add('u','> '+m.text);
    else if(m.type==='think-start')cur=add('think','');
    else if(m.type==='chunk'&&cur)cur.textContent+=m.text;
    else if(m.type==='think-end')cur=null;
    else if(m.type==='act')add('act','▶ '+m.text);
    else if(m.type==='tool')add('tool',m.text);
    else if(m.type==='final'){add('final','✅ '+m.text);lastTeach=teach();}
    else if(m.type==='corrected'&&lastTeach){const n=lastTeach.querySelector('.tnote');if(n)n.textContent='✓ saved — run \`apprentice digest\` to learn it';}
  });
  </script></body></html>`;
}
