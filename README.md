# civ6-rl-lab

Deep reinforcement learning for Civilization VI through the civ6-host client.

The existing `lan_session.collect_session` has now collected four actor-selected
commands from a live GS + BBG 7.5.1 session on the explicit repaired XP2 runtime.
The frozen v16 actor selects production, queue edits and research; native
readbacks confirm their effects. Recurrent state is retained, the public rollout
is recorded, and the session saves and closes normally. These four turn-2
decisions are a bounded diagnostic, with no new optimizer update or strategic
scenario admission. Earlier native actor/learner evidence remains preserved.
The local monitor run is `bbg751-common-session-current092`; private lab receipt:
`evidence/agent-bbg751-common-native-collection-current.json`.

Local Windows worker containment can now receive a caller-owned
`local_process_runtime` alongside `local_memory_limit_bytes`. It must supply
`job(process, limit)`, `memory(handle)` and `kernel.CloseHandle(job)`; invalid
objects are rejected before spawning. This keeps the learner independent of a
GameCore implementation and reuses the existing process controller. The caller
must pair this option with a memory cap. Omitting the runtime keeps the historical
legacy import path; SSH/Mac workers without local Windows containment are
unchanged. The supplied launcher must wait for the existing `go` handshake
before importing Torch, so the process can be assigned to its Job first.

An existing v16 actor can explicitly consume older LAN v24-v31 catalogues;
missing governor status remains unknown. The default encoding stays v15 and
v32 still requires explicit v16. Four scripted native BBG 7.5.1 decisions,
including the official owned-unit swap, pass the frozen v16 actor with every
candidate preserved. This is recurrent scoring of a retained native transcript,
with no checkpoint adaptation, autonomous native selection or PPO update.

This repository is a source snapshot of the PyTorch learner and its monitoring
tools. The model keeps the original `civ6rl.learning` import paths. The legacy
native campaign launchers, optional Windows worker containment, machine-specific
setup and checkpoints still depend on the original local lab workspace.

## Included sources

- Existing map CNN, entity representations and per-player/per-episode LSTM.
- Human-availability action masking, recurrent PPO, GAE/bootstrap and full checkpoint resume.
- Player-only baseline and separate asymmetric critic; privileged training sidecars never enter actor inputs, masks, memory or deployment artifacts.
- Collection, curriculum, held-out evaluation, checkpoints and learning monitor.
- Learner and worker transports, plus the Phase 1 training plan.

The host API and native runtime are maintained in the separate
[`civ6-host` repository](https://github.com/victornu/civ6-host). The snapshot
retains two pure data validators used by the learner's legacy wire formats.
The integration of the learner with the extracted host remains in development.

The LAN v31 envoy extension retains the v15 action layout and all
host candidates. Sending and retaining use the existing action-kind features
and public target-player category. The complete token/target counts remain in
the host catalogue; v15 does not yet encode them as numeric features. Accepting
these actions therefore does not establish a strategically sufficient envoy
representation. Five scripted native decision frames, including retention and
sending with verified effects in Gathering Storm + BBG 7.5.1, pass the frozen
recurrent actor. This is teacher-forced scoring, with no new PPO update or
autonomous native action selection. The exact source-only resumes
are `lan_*envoys_v31*`; retained v30 receipts keep their original identity.

`civ6rl.learning.lan_session.collect_session` is the common collection entry
point for an existing bounded host `SessionClient`. Supply a context-manager
factory that initializes the existing `WorkerClient` with the matching frozen
vocabulary and policy/learner checkpoint. The factory is entered after native
reset, so GameCore loading does not consume the worker's collection wall budget.
The function calls the existing `lan_collection.collect_episode`, forwards policy
card assignments unchanged, closes the session, and returns its original rollout
records for the existing learner. It creates no optimizer and performs no update.

The host's opt-in v32 governor catalogue uses the explicit v16 actor input
schema. Public title/status entities, separate governor and promotion identities,
and all four governor action kinds are represented without dropping candidates.
City assignments retain the exact owner/ID pair; own-city features join on both.
V15 continues to refuse v32. CNN, attention, LSTM and PPO architecture stay fixed;
only the versioned entity/action input projections and vocabulary are extended.
Four scripted native decision frames pass recurrent scoring. Both reference
learner checkpoints have explicit v15-to-v16 adaptations preserving learned
columns, token embeddings, Adam state, critic, RNG and update history. Original
checkpoints and trajectories retain their schemas; old trajectories are not v16
on-policy samples. This is retained-native I/O qualification, with no new PPO
update, autonomous native selection or strategic training admission. Private
receipts: `evidence/host-bbg751-governor-controls-native.json` and
`evidence/agent-bbg751-governor-inputs.json` in the local lab.

A subsequent native reload and three forced turns reach the governor's actual
established state at turn 50. Those three command intervals pass the existing
rollout validator and recurrent scoring with the delivered v16 actor and frozen
vocabulary. The final observation retains all 134 candidates and encodes the
exact native establishment flag and three turns on site. No checkpoint adaptation,
optimizer update or autonomous native selection is repeated for this check.
Private receipt: `evidence/host-bbg751-governor-establishment-native.json`.

The v16 learner also has a bounded optimizer integration fixture: actual parent
Adam moments are adapted, two fresh task-conditioned recurrent choices are
sampled from retained public menus, and a finite PPO update produces nonzero
actor and governor-feature gradients. Its successor states and receipts are
synthetic, so it is software validation rather than native on-policy training.
It changes no retained checkpoint or native training count. Private receipt:
`evidence/phase1-bbg751-integration-source-review.json`.

Each call needs a new private output directory. It streams `exchange.jsonl`
(the existing LAN monitor's input format), `records.jsonl`, a timed `result.json`,
and an atomic `progress.json`. A host/worker failure is recorded as infrastructure
failure and aborts the client; it never becomes a task loss. The worker context
owns policy-process cleanup. A persistent learner can be supplied with
`contextlib.nullcontext`; its caller then owns that process and any subsequent
update. Source-pinned native artifacts and learner checkpoints are unchanged.
Replay tests cover the retained BBG city-control and joint-policy episodes, plus
startup/action/close failures. They do not establish a new native run or strategic
training admission. Native lifecycle/source receipts still need verification
before publishing a completed native run through the existing monitor. Private
critic sidecars stay separate and are not collected into these public logs.

For a public LAN v24-v30 view returned by `civ6_host.session_client.SessionClient`,
install the client package from the host repository alongside this learner.
LAN city focus and citizen catalogues now use the existing v15 features, also
under public display-v2. The adapter passes a narrow transient view to the
legacy strict city indexers; the live v30 schema, candidate IDs and parameters
are preserved. Focus-only and full citizen-query modes are supported. A native
BBG focus command and worked-plot lock replay through recurrent actor scoring
with all 125 candidates, followed by a separate saved-state reload. This is
scripted native I/O qualification, not an autonomous training campaign.
The encoder also accepts the host's explicit French `player-view-display.v2`
projection with a v24-v30 catalogue. Its public science, culture, faith and city
yields populate the existing actor dimensions. Raw LAN views retain unknown
masks for those unqualified values. Candidates, ordering and action masks stay
unchanged. Task collection and recurrent scoring consume the same public view;
the separate critic joins its exact sidecar against the displayed gold floor.
Checkpoint resumes require the reviewed `lan_*public_display_v30*` source
transition; no historical trajectory is relabelled.

This consumer path also replays a real BBG turn-45 display readback through the
common client and frozen actor, retaining all 91 candidates. The host uses a
separately qualified exact BBG runtime/profile; the old stock source closure
continues to refuse changed dependencies. Fixture labels are never substituted
into live collection. This two-frame read/save/close qualification includes no
new gameplay action, policy sample, PPO update or strategic training admission.

Pass the view and a frozen vocabulary from its matching gameplay database to
`civ6rl.learning.encoding.encode_response`. The vocabulary loader accepts an
explicit `gameinfo_evidence_path` and checks its pinned digest. The encoder uses
the v15 actor layout and retains native candidate IDs for dispatch. The Windows
runtime is only required on the machine hosting GameCore.

LAN v25 uses the existing factorized policy-card decoder. The collector forwards
the selected unsigned assignment hashes through the host client, and the rollout
validator joins them to the exact native command receipt. One BBG 7.5.1 free
assignment and its native saved reload have been replayed through collection,
encoding and joint action scoring. The reference v24 completed checkpoints can
resume at a new episode with the explicit source-only migrations
`lan_player_only_policy_assignments_v25` or `lan_policy_assignments_v25` for the
separate privileged critic. These preserve the actor, critic, optimizers and RNG;
retained trajectories keep their original runtime. No strategic learning gain
or complete policy-action coverage follows from this diagnostic.

LAN v26 keeps `policy_unlock` distinct in the host receipt. Its actor
input uses the existing policy-family feature, native cost and yield-currency
token; the CNN, entity encoder, LSTM, heads and vocabulary dimensions stay fixed.
The native BBG 7.5.1 control executes a 55-gold unlock, replaces two policies,
saves and resumes the result in a separate headless process. Its six native
actions replay through the collector and recurrent scoring without dropping
candidates. Both reference checkpoint variants restore their exact learned
state through the existing source-only migration. This qualifies input/output
integration; autonomous selection of the paid action, a new PPO campaign and
playing-strength improvement remain unproven. Private evidence:
`evidence/host-bbg751-native-policy-unlock.json` in the original lab.

LAN v27 also accepts the native government catalogue and the existing
`government`/`retain_government` action features. Native Classical Republic
adoption, four-card assignment, consideration and saved resumes have been replayed
through the client, recurrent scoring and the existing interval validator. Both
checkpoint variants retain their learned state. The no-task context is explicitly
masked during this IO diagnostic; no curriculum, rewards, architecture or optimizer
updates are introduced. The host retains predicted-anarchy rows and declares the
anarchy action path unqualified. Private evidence:
`evidence/host-bbg751-native-government.json` in the original lab.

LAN v28 accepts the host's qualified policy-assignment menu, including native
offers that move an active card to a different slot. The existing factorized
assignment decoder masks duplicate and unchanged complete assignments; actor
and critic dimensions remain unchanged. A synthetic swap projected from a
retained v27 response tests the decoder. The native v28 receipt additionally
swaps two active economic cards at turn45, saves and reloads the same result,
then replays twelve client replies and scores two decisions with the frozen
actor. This adds no PPO update or autonomous relocation-selection witness.
Private evidence: `evidence/host-bbg751-native-policy-relocation.json` in the lab.
The source-only migration `lan_policy_relocation_v28` applies only to its exact
prior source pins. The retained v24 player-only/asymmetric checkpoints also
restore through their existing explicitly pinned v25 migration routes, as
recorded in that receipt. Weights, optimizers and RNG stay unchanged; old
trajectories keep their original source runtime.

The historical changed-only v28 contract omits the UI sequence that removes and
restores a card and confirms the unchanged final assignment. The v29 delivery
below closes that scoped gap without reinterpreting older trajectories.

LAN v29 accepts that complete unchanged assignment only when the fully validated
public view carries the reconfirm schema. The internal `policy_catalog` gate
defaults false for older frames, so v24-v28 keep their changed-only masks. The
factorized decoder, worker receipt, client validation and PPO replay share this
gate without changing actor/critic dimensions or learned parameters. Source-only
transition `lan_policy_reconfirm_v29` and explicit v24 routes
`lan_player_only_policy_reconfirm_v29` / `lan_policy_reconfirm_v29_from_v24`
pin the prior implementation; weights, optimizer state and RNG
are preserved, and retained trajectories keep their original runtime.

The native BBG 7.5.1 witness now confirms the same four cards at turn45:
the engine's confirmation marker changes from false to true, and Save followed
by a fresh-process resume preserves the state and choices. Twelve public replies
replay through the client; two ready frames pass frozen actor scoring and two
native intervals pass the existing rollout validator. Both reference checkpoints
restore with only implementation pins changed. This is scripted native execution
and model I/O qualification, with zero optimizer updates; it does not establish
autonomous confirmation selection or a qualified strategic campaign. Private
evidence: `evidence/host-bbg751-native-policy-reconfirmation.json` in the lab.

The learner now retains those joint hashes through behavior replay, GAE and PPO.
Previously, its LAN branch replaced the assignment with `None`, so a recorded
policy action failed before optimization despite successful host dispatch and
standalone actor scoring. The regression uses the retained native v29 receipts
with explicitly teacher-forced fixture scores: it verifies joint probabilities,
policy-decoder gradients and refusal of assignments that differ from the native
receipt. It does not turn scripted actions into on-policy native training data.
The reference completed checkpoints resume with the explicit transitions
`lan_player_only_joint_policy_learner_v29_from_v24` and
`lan_joint_policy_learner_v29_from_v24`; the reviewed preceding v29 source pins
use `lan_joint_policy_learner_v29`. Learned state is preserved and old trajectories
keep their original runtime. Private evidence:
`evidence/agent-bbg751-lan-joint-policy-learner.json` in the lab.

The single-seat LAN collector and `WorkerClient.update_diagnostic` now share
the existing explicit collection budget. The legacy learner default remains
12 decisions. A larger declared collection, up to the bounded-episode limit of
256, must use the exact budget acknowledged during worker initialization;
the update returns that same contract and permits one update. More than 64
steps reuse the existing checkpointed grouped PPO implementation. Routing to
that upper bound is tested; the retained native regression covers 18 decisions,
not a native 256-decision training campaign.

For an already initialized player-only worker and an admitted scenario, the
collection/update calls are:

```python
summary = collect_episode(
    client, worker, scenario,
    max_decisions=budget["max_decisions"], diagnostic_collection=budget,
)
update = worker.update_diagnostic(summary["records"], scenario, epochs=1)
```

The worker exposes its acknowledged `diagnostic_collection` as a detached copy
and `collection_seconds_remaining` after elapsed work, the update request and
cleanup reserve. The collector refuses a mismatched or exhausted budget.
Decision/time/round limits truncate at an actual observed boundary with a real
successor for bootstrap; they do not change the task deadline or reward.
The budget is recorded in checkpoint provenance, outside actor inputs. The
preceding reviewed runtime resumes through `lan_bounded_episode_v29` at a
completed boundary; historical trajectories retain their original sources.
Legacy Base policy views keep changed-only assignments, while a LAN marker
still requires full LAN validation before enabling reconfirmation semantics.
Private evidence: `evidence/agent-bbg751-lan-episode-budget.json` in the lab.

LAN v30 accepts separate gold and faith unit/building purchase quotes through
the existing currency/cost features, vocabulary and action masks. It preserves
each published candidate ID, including two currencies for the same item, and
inherits validated v29 policy reconfirmation. Actor and critic dimensions are
unchanged. `lan_faith_purchases_v30` resumes the exact preceding source pins at
a completed learner boundary; older trajectory runtimes are not rewritten.
The tests add faith offers as explicit fixtures to a retained native v29 view.
A positive native XP2 faith-purchase effect remains unqualified and is reported
by the host family; this is not admission of a strategic training campaign.
Private evidence: `evidence/host-xp2-faith-purchases-api.json` in the lab.

One actual native v30 BBG turn45 query has also passed `SessionClient` replay
and frozen actor scoring with all 91 candidate IDs intact. Both reference
learner checkpoints restore with only their reviewed implementation pins
changed. This is a readback test with no sampled action, turn or optimizer
update; the fourteen native city quotes at that boundary contain no faith
offer. Private evidence: `evidence/host-bbg751-v30-native-query.json` in the lab.

The current LAN qualification covers the offers observed in one BBG 7.5.1
development scenario. In a bounded live diagnostic, the policy executed 23
actions through `SessionClient` and advanced from turn 2 to turn 5 with its
recurrent state retained. The sampling seed was selected to exercise the
turn path. The diagnostic stopped at its collection limit. Unpublished map cells
and unqualified display precision stay unknown. An unsupported choice causes
the encoder to reject the whole menu. The full LAN action inventory remains
open.

The bounded player-only learner accepts LAN diagnostic records through
`update_diagnostic`. To produce them, call
`civ6rl.learning.lan_collection.collect_episode(client, worker, scenario)`
with an already reset single-human LAN client. The caller supervises the
session and closes it. The collector stops at an observed task result or its
diagnostic limit and raises if it cannot obtain a usable successor.
The default cap is 12 decisions. Evaluation may explicitly request up to 64
with `max_decisions`, and may set `max_collection_seconds` to stop at the
next complete interval after that time. Keep a hard native lifetime limit in
the supervisor. The PPO diagnostic still accepts at most 12 decisions.
Each record contains `before`, the worker's `choice`,
the `command_receipt` from `SessionClient.last_command_receipt`, and
`successors` in order from the action response through any polls. Retain
the same player's published views and stop the interval at its next ready
decision or observed terminal boundary. The update checks native receipts,
replays recurrent behavior and uses the existing PPO and checkpoint pipeline.
The LAN checkpoint covers this diagnostic route and pins its source files.
The exact BBG 7.5.1 diagnostic can opt into a separate private critic trace
via `NetworkBackend(privileged_critic_diagnostic=True, ...)`. Its ready-state
records can be joined offline with
`civ6rl.learning.lan_critic.encode_lan_privileged_snapshot`. Supply the pinned
Core, host, closure and collector hashes independently. The bounded LAN PPO
diagnostic accepts these separate records in `privileged_diagnostic` mode,
with a snapshot for each decision and a final successor for bootstrap.
The original player-only checkpoint can be read through
`load_training_bytes(..., implementation_migration="lan_critic_sidecar_v1")`.
That source-only migration preserves its actor, optimizer and critic mode.
Strategic collection and simultaneous self-play still require qualification.

To install the source and run the included tests:

```shell
python -m pip install -e .
python -m unittest discover -s tests
```

Game binaries, official captures and databases, private traces, credentials and
model weights are stored separately from this source snapshot.
