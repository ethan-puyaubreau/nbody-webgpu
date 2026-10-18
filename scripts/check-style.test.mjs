import { expect, test } from "vitest";
import { checkCommit, findViolations } from "./check-style.mjs";

test("findViolations flags banned text and ignores code", () => {
  const text = "It's just a colour — we agree.\n```\njust code\n```\n";
  const rules = findViolations(text, "markdown").map((v) => v.rule);
  expect(rules.sort()).toEqual(["banned-word", "colour", "contraction", "em-dash", "we"]);
  expect(findViolations("const just = 1; // fine here\n", "ts")).toEqual([]);
  expect(findViolations("const a = 1; // it's odd\n", "ts")).toHaveLength(1);
});

test("checkCommit enforces subject, trailer and body rules", () => {
  const good = "fix: never report a lower peak than before";
  expect(checkCommit({ subject: good })).toEqual([]);
  expect(checkCommit({ subject: "chore: start the repository", isRoot: true })).toEqual([]);
  expect(checkCommit({ subject: "chore: start the repository" })).toHaveLength(1);
  expect(checkCommit({ subject: "Fixed stuff." }).length).toBeGreaterThan(1);
  expect(checkCommit({ subject: good, body: "Co-Authored-By: someone" })).toHaveLength(1);
  expect(checkCommit({ subject: good, body: "x".repeat(73) })).toHaveLength(1);
});
