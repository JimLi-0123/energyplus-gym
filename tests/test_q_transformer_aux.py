"""CPU checks without starting EnergyPlus or allocating the production replay buffer."""
import importlib.util
import copy
import csv
import itertools
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AuxiliaryPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Isolate model tests from the environment's optional simulator imports.
        cls.modules = patch.dict(sys.modules)
        cls.modules.start()
        for name in ("eplus_gym", "eplus_gym.agents", "eplus_gym.agents.q_transformer"):
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        base = "src/eplus_gym/agents/q_transformer/"
        prefix = "eplus_gym.agents.q_transformer."
        load_module(prefix + "positional_encoding", base + "positional_encoding.py")
        cls.model = load_module(prefix + "transformer_model", base + "transformer_model.py")
        cls.network = load_module(prefix + "deep_q_network", base + "deep_q_network.py")
        load_module(prefix + "replay_memory", base + "replay_memory.py")
        cls.diagnostics = load_module(prefix + "gradient_diagnostics", base + "gradient_diagnostics.py")
        preprocessing = types.ModuleType("sklearn.preprocessing")
        preprocessing.MinMaxScaler = Mock()
        environment = types.ModuleType("eplus_gym.envs.energyplus")
        environment._find_project_root = lambda: ROOT
        sys.modules["sklearn.preprocessing"] = preprocessing
        sys.modules["eplus_gym.envs.energyplus"] = environment
        cls.agent = load_module(prefix + "dqn_agent", base + "dqn_agent.py")
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.modules.stop()

    def make_model(self, co2=True, iat=True, occ=False):
        return self.model.TransformerModel(
            n_actions=3, d_model=8, nhead=2, num_enc_layers=1,
            num_dec_layers=1, dim_ff=16, max_len=4, use_gate=False,
            use_co2_aux=co2, use_iat_aux=iat, use_occ_aux=occ,
        )

    def test_switches_action_conditioning_and_gradients(self):
        states = torch.randn(2, 4, 18)
        for co2 in (False, True):
            for iat in (False, True):
                with self.subTest(co2=co2, iat=iat):
                    torch.manual_seed(101)
                    model = self.make_model(co2, iat).eval()
                    q, predictions = model(states, return_aux=True, actions=torch.zeros(2, dtype=torch.long))
                    q_other, other = model(states, return_aux=True, actions=torch.ones(2, dtype=torch.long))
                    torch.testing.assert_close(q, q_other)
                    torch.testing.assert_close(q, model(states))
                    expected = ({"co2"} if co2 else set()) | ({"iat"} if iat else set())
                    self.assertEqual(set(predictions), expected)
                    for key in expected:
                        self.assertEqual(predictions[key].shape, (2, 1))
                        self.assertFalse(torch.allclose(predictions[key], other[key]))
                    loss = sum((p.square().mean() for p in predictions.values()), q.square().mean())
                    loss.backward()
                    self.assertEqual(model.co2_head[0].weight.grad is not None, co2)
                    self.assertEqual(model.iat_head[0].weight.grad is not None, iat)
                    if expected:
                        self.assertIsNotNone(model.aux_action_embedding.weight.grad)
                        self.assertIsNotNone(model.in_embed.weight.grad)

    def test_active_prediction_requires_action(self):
        with self.assertRaisesRegex(ValueError, "executed actions"):
            self.make_model()(torch.zeros(2, 4, 18), return_aux=True)

    def test_occupancy_switch_and_action_independence(self):
        states = torch.randn(2, 4, 18)
        for co2, iat, occ in itertools.product((False, True), repeat=3):
            with self.subTest(co2=co2, iat=iat, occ=occ):
                model = self.make_model(co2, iat, occ).eval()
                q, predictions = model(states, return_aux=True, actions=torch.zeros(2, dtype=torch.long))
                q_other, other = model(states, return_aux=True, actions=torch.ones(2, dtype=torch.long))
                torch.testing.assert_close(q, q_other)
                self.assertEqual("occ" in predictions, occ)
                if occ:
                    torch.testing.assert_close(predictions["occ"], other["occ"])
                    self.assertEqual(predictions["occ"].shape, (2, 1))
                    predictions["occ"].square().mean().backward()
                    self.assertIsNotNone(model.occ_head[0].weight.grad)
                    self.assertGreater(model.in_embed.weight.grad.abs().sum().item(), 0)
                    self.assertIsNone(model.aux_action_embedding.weight.grad)
        model = self.make_model(False, False, True)
        self.assertIn("occ", model(states, return_aux=True)[1])

    def test_learn_uses_executed_actions_and_next_targets(self):
        flags = (False, True)
        for co2, iat, occ, weight, delta in itertools.product(flags, flags, flags, (0.0, .1), flags):
            with self.subTest(co2=co2, iat=iat, occ=occ, weight=weight, delta=delta):
                agent = self.agent.Q_transformer.__new__(self.agent.Q_transformer)
                agent.batch_size = 2
                agent.memory = Mock(t_mem_cntr=2, transformer_mem_size=2)
                current = np.zeros((2, 4, 18), dtype=np.float32)
                current[:, -1, 7] = [.25, .75]
                following = current.copy()
                following[:, -1, 14] = .7
                following[:, -1, 13] = .3
                following[:, -1, 7] = .5
                agent.memory.sample_transformer_buffer.return_value = (
                    current, np.array([[1], [2]]), following,
                    np.zeros(2, dtype=np.float32), np.zeros(2, dtype=bool),
                )
                agent.scaler = Mock(transform=lambda x: x)
                agent.co2_feature_indices = (14,)
                agent.iat_feature_index = 13
                agent.occ_feature_index = 7
                agent.occ_predict_delta = delta
                agent.use_occ_aux = occ
                agent.occ_aux_weight = weight
                agent.use_co2_aux, agent.use_iat_aux = co2, iat
                agent.co2_aux_weight = agent.iat_aux_weight = weight
                agent.gamma, agent.learn_step_counter = .99, 0
                agent.record_gradient_diagnostics = True
                agent.replace_target_network = Mock()
                agent.q_eval = self.network.DeepQNetwork(
                    lr=.001, n_actions=3, name="test", input_dims=(18,),
                    chkpt_dir=".", d_model=8, max_len=4, use_gate=False,
                    use_co2_aux=co2 and weight > 0, use_iat_aux=iat and weight > 0,
                    use_occ_aux=occ and weight > 0,
                ).cpu()
                agent.q_eval.device = torch.device("cpu")
                agent.q_next = agent.q_eval
                calls = []
                agent.q_eval.loss.forward = Mock(side_effect=lambda pred, target: calls.append((pred.detach(), target.detach())) or (pred - target).square().mean())
                actions_seen = []
                handle = agent.q_eval.transformer.aux_action_embedding.register_forward_pre_hook(
                    lambda module, args: actions_seen.append(args[0].clone())
                )
                agent.learn()
                handle.remove()
                targets = [target for _, target in calls[1:]]
                if co2 and weight > 0:
                    torch.testing.assert_close(targets.pop(0), torch.full((2, 1), .7))
                if iat and weight > 0:
                    torch.testing.assert_close(targets.pop(0), torch.full((2, 1), .3))
                if occ and weight > 0:
                    target = torch.tensor([[.25], [-.25]]) if delta else torch.full((2, 1), .5)
                    torch.testing.assert_close(targets.pop(0), target)
                    stats = agent.last_gradient_diagnostics
                    self.assertEqual(stats['occ_changed_count'], 2)
                    self.assertAlmostEqual(stats['occ_prediction_mse'], agent.last_occ_aux_loss, places=5)
                    self.assertAlmostEqual(stats['occ_persistence_mse'], .0625)
                self.assertEqual(targets, [])
                self.assertEqual(len(actions_seen), int((co2 or iat) and weight > 0))
                if actions_seen:
                    torch.testing.assert_close(actions_seen[0], torch.tensor([1, 2]))
                self.assertEqual(agent.learn_step_counter, 1)
                if (co2 or iat or occ) and weight > 0:
                    self.assertIsNotNone(agent.last_gradient_diagnostics)
                    self.assertAlmostEqual(agent.last_gradient_diagnostics['weighted_aux_loss'],
                                           weight * sum((p - t).square().mean().item() for p, t in calls[1:]), places=5)
                else:
                    self.assertIsNone(agent.last_gradient_diagnostics)
                if not (co2 and weight > 0):
                    self.assertEqual(agent.last_co2_aux_loss, 0)
                if not (iat and weight > 0):
                    self.assertEqual(agent.last_iat_aux_loss, 0)
                if not (occ and weight > 0):
                    self.assertEqual(agent.last_occ_aux_loss, 0)

    def test_legacy_checkpoint_preserves_q_values(self):
        with tempfile.TemporaryDirectory() as directory:
            network = self.network.DeepQNetwork(
                lr=.001, n_actions=3, name="legacy", input_dims=(18,),
                chkpt_dir=directory, d_model=8, max_len=4, use_gate=False,
            )
            states = torch.randn(2, 4, 18, device=network.device)
            expected_q = network(states).detach()
            legacy = network.state_dict()
            key = "transformer.co2_head.0.weight"
            legacy[key] = legacy[key][:, :8].clone()
            legacy = {k: v for k, v in legacy.items() if not k.startswith(("transformer.iat_head.", "transformer.occ_head.", "transformer.aux_action_embedding."))}
            torch.save(legacy, network.checkpoint_file)
            network.load_checkpoint()
            torch.testing.assert_close(network(states), expected_q)
            self.assertEqual(torch.count_nonzero(network.state_dict()[key][:, 8:]).item(), 0)

    def test_gradient_geometry_and_unchanged_update(self):
        for direction, expected_cosine in ((-1, -1), (1, 1), (0, float('nan'))):
            parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
            unused = torch.nn.Parameter(torch.ones(1))
            q_loss = parameter.square().sum()
            auxiliary = direction * parameter.square().sum()
            metrics = self.diagnostics.compare_loss_gradients(q_loss, .01 * auxiliary, [parameter, unused])
            self.assertAlmostEqual(metrics['aux_to_q_grad_ratio'], .01 * abs(direction), places=6)
            self.assertAlmostEqual(metrics['q_encoder_grad_norm'], 20 ** .5, places=6)
            self.assertIsNone(parameter.grad)
            if direction:
                self.assertAlmostEqual(metrics['q_aux_grad_cosine'], expected_cosine)
                self.assertEqual(metrics['gradient_conflict'], direction < 0)
            else:
                self.assertFalse(metrics['gradient_direction_valid'])
                self.assertTrue(np.isnan(metrics['q_aux_grad_cosine']))
                self.assertIsNone(metrics['gradient_conflict'])
            (q_loss + .01 * auxiliary).backward()
            torch.testing.assert_close(parameter.grad, (1 + .01 * direction) * 2 * parameter.detach())
        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        metrics = self.diagnostics.compare_loss_gradients(parameter[0], parameter[1], [parameter])
        self.assertEqual(metrics['q_aux_grad_cosine'], 0.)
        metrics = self.diagnostics.compare_loss_gradients((parameter * 0).sum(), parameter.sum(), [parameter])
        self.assertTrue(np.isnan(metrics['aux_to_q_grad_ratio']))

        torch.manual_seed(101)
        model = self.make_model(False, False, True)
        reference = copy.deepcopy(model)
        states = torch.randn(2, 4, 18)
        for network, measure in ((model, True), (reference, False)):
            optimizer = torch.optim.Adam(network.parameters(), lr=.001)
            q, aux = network(states, return_aux=True)
            q_loss, occ_loss = q.square().mean(), aux['occ'].square().mean()
            if measure:
                parameters = [p for module in (network.in_embed, network.encoder, network.mem_norm)
                              for p in module.parameters()]
                self.diagnostics.compare_loss_gradients(q_loss, .01 * occ_loss, parameters)
            (q_loss + .01 * occ_loss).backward()
            optimizer.step()
        for value, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_summary_excludes_undefined_directions(self):
        analyzer = load_module('analyse_aux_gradients', 'tools/analyse_aux_gradients.py')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diagnostics.csv'
            row = dict(seed=101, episode=1, q_loss=1, occ_aux_loss=2,
                       weighted_aux_loss=.02, total_loss=1.02,
                       q_encoder_grad_norm=1, occ_encoder_grad_norm=2,
                       weighted_aux_encoder_grad_norm=.02, aux_to_q_grad_ratio=.02,
                       q_aux_grad_cosine=-.5, gradient_direction_valid=True)
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
                writer.writerow(dict(row, q_aux_grad_cosine=.5))
                writer.writerow(dict(row, q_aux_grad_cosine=float('nan'),
                                     gradient_direction_valid=False, aux_to_q_grad_ratio=float('nan')))
            summary = analyzer.summarize([path])[0]
            self.assertEqual(summary['updates'], 3)
            self.assertEqual(summary['valid_directions'], 2)
            self.assertEqual(summary['conflict_percent'], 50)
            self.assertEqual(summary['cosine_mean'], 0)
            self.assertEqual(summary['ratio_mean'], .02)

    def test_occupancy_persistence_metrics(self):
        target = torch.tensor([[0.], [1.], [.5]])
        current = torch.tensor([[0.], [0.], [.5]])
        prediction = torch.tensor([[.2], [.8], [.5]], requires_grad=True)
        stats = self.diagnostics.occupancy_prediction_metrics(prediction, target, current)
        self.assertEqual(stats['occ_sample_count'], 3)
        self.assertEqual(stats['occ_changed_count'], 1)
        self.assertAlmostEqual(stats['occ_prediction_mse'], .08/3, places=6)
        self.assertAlmostEqual(stats['occ_persistence_mse'], 1/3, places=6)
        self.assertAlmostEqual(stats['occ_changed_prediction_sse'], .04, places=6)
        self.assertEqual(stats['occ_changed_persistence_sse'], 1.)
        self.assertIsNone(prediction.grad)
        steady = self.diagnostics.occupancy_prediction_metrics(prediction, target, target)
        self.assertEqual(steady['occ_changed_count'], 0)
        self.assertEqual(steady['occ_changed_prediction_sse'], 0)
        self.assertTrue(np.isnan(steady['occ_changed_prediction_mse']))


if __name__ == "__main__":
    unittest.main()
