"""v27 government offers use existing actor dimensions and public identities."""
from copy import deepcopy
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import unittest

from civ6_host.contracts import government, network_turn
from civ6_host.runtime.paths import resource_root
from civ6rl.learning.content_vocabulary import load_public_content_vocabulary
from civ6rl.learning.encoding import ACTOR_V15, actor_feature_names, encode_response


class GovernmentInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=resource_root()
        replay=root/'work/bbg751-policy-unlock/payment/public-replay.jsonl'
        capture=root/'work/official-capture/xp2-bbg751-correlated-20261104/recovered-exact-inputs.json'
        if not replay.is_file() or not capture.is_file():
            raise unittest.SkipTest('Retained native v26 reply and pinned public database required')
        with replay.open('rb') as stream:
            recent=deque((json.loads(line) for line in stream),maxlen=2)
        cls.view=recent[0]['view']
        db=json.loads(capture.read_bytes())['database_copies']['gameplay']
        db_path=Path(db['path'])
        assert sha256(db_path.read_bytes()).hexdigest()==db['sha256']
        cls.vocab=load_public_content_vocabulary(db_path,expected_sha256=db['sha256'],
            gameinfo_evidence_path=root/'evidence/phase41-gameinfo-maps.json',include_action_identities=True)
        with sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True) as connection:
            cls.gov_hashes={name:code&0xffffffff for name,code in connection.execute(
                'SELECT GovernmentType,Hash FROM Governments JOIN Types ON GovernmentType=Type')}

    def v27(self):
        view=deepcopy(self.view)
        menu=view['decisions']
        menu['schema']=network_turn.CAPABILITY_XP2_GOVERNMENT_SCHEMA
        menu['government']=dict(player_id=view['observation']['player_id'],turn=view['observation']['turn'],
            current_index=0,can_change=True,changed=False,civic_completed=True,
            considered=False,consideration_blocker=True,choices=[
                dict(index=0,hash=self.gov_hashes['GOVERNMENT_CHIEFDOM'],unlocked=True,can_change=False,anarchy_turns=0),
                dict(index=3,hash=self.gov_hashes['GOVERNMENT_CLASSICAL_REPUBLIC'],unlocked=True,can_change=True,anarchy_turns=0)])
        choices=government.actions(menu['government'])
        rows=[dict(candidate_id=network_turn.digest(a),action=a) for a in choices]
        menu['candidates'].extend(rows)
        menu['families']['government']=dict(status='queried_partial',
            candidate_ids=[row['candidate_id'] for row in rows],unqualified=['anarchy_transition'])
        menu.pop('decision_id');menu['decision_id']=network_turn.digest(menu)
        return view,rows

    def test_retained_native_layout_with_synthetic_government_offers_preserves_all_candidates(self):
        view,rows=self.v27()
        frame=encode_response(view,vocabulary=self.vocab,actor_schema=ACTOR_V15)
        _,features=actor_feature_names(ACTOR_V15)
        self.assertEqual(frame['candidate_ids'],[r['candidate_id'] for r in view['decisions']['candidates']])
        self.assertTrue(all(frame['action_mask']))
        for row in rows:
            i=frame['candidate_ids'].index(row['candidate_id'])
            self.assertEqual(len(frame['candidates'][i]),len(features))
            self.assertEqual(frame['candidates'][i][features.index('kind:'+row['action']['kind'])],1.0)

    def test_privileged_extras_in_government_channel_are_refused(self):
        view,rows=self.v27()
        view['decisions']['government']['enemy_government_hidden']=3
        view['decisions'].pop('decision_id')
        view['decisions']['decision_id']=network_turn.digest(view['decisions'])
        with self.assertRaises(ValueError): encode_response(view,vocabulary=self.vocab,actor_schema=ACTOR_V15)
