// Encodes the writing standards of the project as exit codes.
// Usage: node scripts/check-style.mjs [--commits [range]]
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const RULES = [
  ["em-dash", /—/g],
  ["emoji", /\p{Emoji_Presentation}|️/gu],
  ["colour", /colour/gi],
  ["we", /\bwe\b/gi],
  ["contraction", /n['’]t\b|\b(?:it|there|that|what|let)['’]s\b|['’](?:re|ll|ve)\b/gi],
  ["banned-word", /\b(?:should|might|simply|easily|powerful|just|roughly)\b/gi],
];

const PREFIXES = ["feat", "fix", "docs", "test", "ci", "perf", "build", "bench", "chore", "refactor"];
const SUBJECT_MIN = 35; // applies to non-root commits only: the root commit "chore: start the repository" has 27
const SUBJECT_MAX = 65;
const BODY_MAX = 72;

// Replace every character that is not a newline with a space, so line numbers survive.
const blank = (s) => s.replace(/[^\n]/g, " ");

function stripMarkdownFences(text) {
  return text.replace(/^(```|~~~)[^\n]*\n[\s\S]*?^\1[^\n]*$/gm, blank);
}

function stripHtmlMarkup(text) {
  return text.replace(/<(script|style)\b[\s\S]*?<\/\1>/gi, blank).replace(/<[^>]*>/g, (tag) => {
    const description = /^<meta\b(?=[^>]*\bname=["']description["'])[^>]*?\bcontent=(["'])([\s\S]*?)\1/i.exec(tag);
    if (!description) return blank(tag);
    const start = tag.indexOf(description[2], tag.indexOf("content"));
    return blank(tag.slice(0, start)) + description[2] + blank(tag.slice(start + description[2].length));
  });
}

// Keeps only comments (and, with strings: true, string literal contents) of a source file.
function keepCode(text, { lineComments, strings }) {
  let out = "";
  let i = 0;
  const emit = (chunk, keep) => {
    out += keep ? chunk : blank(chunk);
  };
  while (i < text.length) {
    const two = text.slice(i, i + 2);
    if (two === "/*") {
      const end = text.indexOf("*/", i + 2);
      const stop = end === -1 ? text.length : end + 2;
      emit(text.slice(i, stop), true);
      i = stop;
    } else if (lineComments && two === "//") {
      const end = text.indexOf("\n", i);
      const stop = end === -1 ? text.length : end;
      emit(text.slice(i, stop), true);
      i = stop;
    } else if (strings && /["'`]/.test(text[i])) {
      const quote = text[i];
      let j = i + 1;
      while (j < text.length && text[j] !== quote && (quote === "`" || text[j] !== "\n")) {
        j += text[j] === "\\" ? 2 : 1;
      }
      emit(text[i], false);
      emit(text.slice(i + 1, j), true);
      emit(text.slice(j, j + 1), false);
      i = j + 1;
    } else {
      emit(text[i], false);
      i += 1;
    }
  }
  return out;
}

// kind: "markdown" | "html" | "css" | "wgsl" | "ts" | "main" (ts comments plus string literals)
export function findViolations(text, kind) {
  const scoped = {
    markdown: () => stripMarkdownFences(text),
    html: () => stripHtmlMarkup(text),
    css: () => keepCode(text, { lineComments: false, strings: false }),
    wgsl: () => keepCode(text, { lineComments: true, strings: false }),
    ts: () => keepCode(text, { lineComments: true, strings: false }),
    main: () => keepCode(text, { lineComments: true, strings: true }),
  }[kind]();
  const found = [];
  scoped.split("\n").forEach((line, index) => {
    for (const [rule, pattern] of RULES) {
      for (const match of line.matchAll(pattern)) found.push({ line: index + 1, rule, match: match[0] });
    }
  });
  return found;
}

// Returns a list of problems; an empty list means the commit follows the standard.
export function checkCommit({ subject, body = "", isRoot = false }) {
  const problems = [];
  const parsed = new RegExp(`^(${PREFIXES.join("|")})(\\([^)]+\\))?: (.*)$`).exec(subject);
  if (!parsed) problems.push(`subject must start with one of ${PREFIXES.join(", ")} followed by ": "`);
  else if (!/^[a-z]/.test(parsed[3])) problems.push("first letter after the colon must be lower case");
  if (subject.length > SUBJECT_MAX)
    problems.push(`subject has ${subject.length} characters, maximum is ${SUBJECT_MAX}`);
  if (subject.length < SUBJECT_MIN && !isRoot) {
    problems.push(`subject has ${subject.length} characters, minimum is ${SUBJECT_MIN}`);
  }
  if (subject.endsWith(".")) problems.push("subject must not end with a period");
  if (/^co-authored-by:/im.test(body)) problems.push("Co-Authored-By trailer is not allowed");
  for (const line of body.split("\n")) {
    if (line.length > BODY_MAX) problems.push(`body line has ${line.length} characters, maximum is ${BODY_MAX}`);
  }
  return problems;
}

function listFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    if (entry.name === "node_modules" || entry.name === "dist") return [];
    const path = join(dir, entry.name);
    return entry.isDirectory() ? listFiles(path) : [path];
  });
}

function kindOf(path) {
  const normalized = path.replaceAll("\\", "/");
  if (normalized === "src/main.ts") return "main";
  if (normalized.endsWith(".ts") || normalized.endsWith(".mjs")) return "ts";
  if (normalized.endsWith(".wgsl")) return "wgsl";
  if (normalized.endsWith(".css")) return "css";
  return null;
}

function runTextChecks() {
  const targets = [];
  for (const name of ["README.md", "CHANGELOG.md"]) if (existsSync(name)) targets.push([name, "markdown"]);
  if (existsSync("index.html")) targets.push(["index.html", "html"]);
  for (const name of ["vite.config.ts"]) if (existsSync(name) && kindOf(name)) targets.push([name, kindOf(name)]);
  for (const file of [...listFiles("src"), ...listFiles("scripts")]) {
    const kind = file.endsWith(".test.mjs") ? null : kindOf(file); // test files hold violations as fixtures
    if (kind) targets.push([file, kind]);
  }
  let count = 0;
  for (const [file, kind] of targets) {
    for (const v of findViolations(readFileSync(file, "utf8"), kind)) {
      console.log(`${file.replaceAll("\\", "/")}:${v.line}:${v.rule} (${v.match})`);
      count += 1;
    }
  }
  return count;
}

function git(...args) {
  return execFileSync("git", args, { encoding: "utf8" });
}

function runCommitChecks(range) {
  let target = range;
  if (!target) {
    try {
      git("rev-parse", "--verify", "--quiet", "origin/main");
      target = "origin/main..HEAD";
    } catch {
      target = "HEAD";
    }
  }
  const log = git("log", "--format=%h%x1f%P%x1f%s%x1f%b%x1e", target);
  let count = 0;
  for (const record of log.split("\x1e")) {
    if (!record.trim()) continue;
    const [hash, parents, subject, body] = record.trim().split("\x1f");
    for (const problem of checkCommit({ subject, body: body ?? "", isRoot: parents === "" })) {
      console.log(`${hash}:${problem}`);
      count += 1;
    }
  }
  return count;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const count = args[0] === "--commits" ? runCommitChecks(args[1]) : runTextChecks();
  process.exit(count > 0 ? 1 : 0);
}
