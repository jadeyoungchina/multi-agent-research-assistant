# Reconstruction history

This repository was reconstructed on 2026-09-08 after the original local project was accidentally deleted. Commit dates restore the documented development milestones; they are not the original Git objects or an unreconstructed historical record.

本仓库于 2026-09-08 在原本地项目意外删除后重建。提交日期用于恢复已记录的开发里程碑；这些提交不是原始 Git 对象，也不是未经重建的历史记录。

The documented reconstruction timeline represents development milestones, not
forensic recovery of deleted commits. Code, tests, and contemporary verification
are the evidence for the delivered behavior. Implementation and verification
can take place after the nominal milestone date; neither an authored date nor
this document proves publication at that time.

The approved design records this target timeline in Asia/Shanghai time:

| Milestone date/time | Scope |
| --- | --- |
| 2026-07-06 20:18 | Environment, license, upstream analysis |
| 2026-07-13 21:07 | Skeleton, settings, domain models |
| 2026-07-22 22:16 | Agents and graph |
| 2026-07-30 20:43 | Document intake and chunks |
| 2026-08-08 21:26 | Retrieval and citation tracing |
| 2026-08-17 22:09 | Critic revision and recovery |
| 2026-08-25 20:51 | Qwen/OpenAI providers and API |
| 2026-09-01 21:34 | Evaluation framework and fixtures |
| 2026-09-05 22:12 | Browser UI and integration |
| 2026-09-08 21:40 | Verification, documentation, v1.0.0 milestone |

This table is the reconstruction plan. Consult the actual Git log and remote
tags to establish which commits and releases have been published. The release
checklist does not mark remote publication complete merely because local docs
or tests exist.

Upstream references are pinned to `4c95ae14cc2462c442b5c064cccd74430d02bc46`.
The audit records source availability dates, including 2026-07-03 for document
intake and 2026-08-28 for trace evaluation; reconstructed attribution must not
predate source availability. Planned mappings in older design/plan files may
differ from the final implementation; [third-party notices](../THIRD_PARTY_NOTICES.md)
and [upstream analysis](upstream-analysis.md) describe the delivered mapping.

Historical environment notes (`environment-setup.md`, `environment-check.txt`)
record an earlier setup snapshot. Their pending credentials, Git identity, and
next-step notes are not current operating instructions; use the README.

Any authorized history rewrite must retain a local bundle backup, verify the
final tree and tests, and use an expected remote lease rather than a blind force
push. Publication and tagging are separate release actions, not performed by
the documentation verifier or CI.
