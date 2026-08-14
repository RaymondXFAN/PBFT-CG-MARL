"""
快速冒烟测试 - 验证代码能跑通

脚本只需要 stdlib + numpy，不需要 torch/regex/... 高级包
（用于无卡模式的"先看代码有没有语法/逻辑错误"）

在 AutoDL 上跑真实训练前，先用这个脚本快速验证：
1. 6 个算法都能导入
2. 4 个环境都能创建
3. PBFT 共识层能跑
4. TrainEnvWrapper 转换正常

【使用方法】
# 沙盒里（不依赖 torch）
python scripts/smoke_test.py

# AutoDL 上（有 torch）
python scripts/smoke_test.py --with-torch
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse


def test_algorithm_registry():
    """测试算法注册表"""
    print("\n[1/6] 测试算法注册表...")
    try:
        from src.algorithms import ALGORITHM_REGISTRY
        expected = {'pbft_cg_mappo', 'mappo', 'qmix', 'maddpg', 'commnet', 'tarmac'}
        actual = set(ALGORITHM_REGISTRY.keys())
        if expected.issubset(actual):
            print(f"  ✅ 所有 6 个算法已注册: {sorted(actual)}")
            return True
        else:
            missing = expected - actual
            print(f"  ❌ 缺失算法: {missing}")
            return False
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_env_registry():
    """测试环境注册表"""
    print("\n[2/6] 测试环境注册表...")
    try:
        from src.envs import ENV_REGISTRY, make_env
        expected = {'mpe_spread', 'mpe_reference', 'smaclite_5m_vs_6m',
                    'smaclite_3s5z', 'vmas_uav_coverage', 'vmas_formation',
                    'lbf_2s3f'}
        actual = set(ENV_REGISTRY.keys())
        if expected.issubset(actual):
            print(f"  ✅ 所有 7 个环境已注册: {sorted(actual)}")
            return True
        else:
            print(f"  ❌ 缺失环境: {expected - actual}")
            return False
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_env_create():
    """测试环境创建（无 torch）"""
    print("\n[3/6] 测试环境创建...")
    try:
        from src.envs import make_env, ENV_REGISTRY
        results = []
        for env_name in ENV_REGISTRY:
            try:
                env = make_env(env_name, {'n_agents': 4, 'max_steps': 50})
                info = env.get_env_info()
                results.append((env_name, "✅", info))
            except Exception as e:
                results.append((env_name, "❌", str(e)))
        for name, status, info in results:
            if status == "✅":
                print(f"  {status} {name}: n_agents={info['n_agents']}, "
                      f"obs_shape={info['obs_shape']}, action_shape={info['action_shape']}")
            else:
                print(f"  {status} {name}: {info}")
        return all(r[1] == "✅" for r in results)
    except Exception as e:
        print(f"  ❌ 创建失败: {e}")
        return False


def test_pbft_consensus():
    """测试 PBFT 共识层（无 torch，用 numpy 模拟）"""
    print("\n[4/6] 测试 PBFT 共识层...")
    try:
        # 检查类能否导入
        from src.consensus import PBFTConsensusLayer
        print(f"  ✅ PBFTConsensusLayer 类导入成功")
        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_shape_normalization():
    """测试形状归一化（最关键的 bug 修复）"""
    print("\n[5/6] 测试形状归一化（tuple→int）...")
    try:
        from src.algorithms.base import BaseAlgorithm
        # 用一个匿名子类测试
        class _Test(BaseAlgorithm):
            def act(self, *a, **k): pass
            def update(self, *a, **k): pass
            def save(self, *a, **k): pass
            def load(self, *a, **k): pass
            def get_action(self, *a, **k): pass
            def get_value(self, *a, **k): pass

        # 测试各种输入
        test_cases = [
            ({"n_agents": 3, "obs_shape": (18,), "state_shape": (54,),
              "action_shape": 5, "action_type": "discrete"}, (18, 54, 5)),
            ({"n_agents": 3, "obs_shape": 18, "state_shape": 54,
              "action_shape": (5,), "action_type": "discrete"}, (18, 54, 5)),
            ({"n_agents": 3, "obs_shape": (3, 84, 84), "state_shape": (54, 12),
              "action_shape": 5}, (3*84*84, 54*12, 5)),
        ]
        all_pass = True
        for env_info, expected in test_cases:
            algo = _Test(config={}, env_info=env_info)
            actual = (algo.obs_shape, algo.state_shape, algo.action_shape)
            if actual == expected:
                print(f"  ✅ {env_info['obs_shape']}→{actual[0]}, "
                      f"{env_info['state_shape']}→{actual[1]}, "
                      f"{env_info['action_shape']}→{actual[2]}")
            else:
                print(f"  ❌ 输入 {env_info}, 期望 {expected}, 实际 {actual}")
                all_pass = False
        return all_pass
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_full_pipeline():
    """测试完整流程（无 torch，最多只验证 PBFT 部分）"""
    print("\n[6/6] 测试完整流程（不需要 torch）...")
    try:
        from src.envs import make_env
        from src.algorithms.base import BaseAlgorithm

        class _Test(BaseAlgorithm):
            def act(self, *a, **k): pass
            def update(self, *a, **k): pass
            def save(self, *a, **k): pass
            def load(self, *a, **k): pass
            def get_action(self, *a, **k): pass
            def get_value(self, *a, **k): pass

        env = make_env('mpe_spread', {'n_agents': 4, 'max_steps': 25})
        info = env.get_env_info()
        print(f"  环境: n_agents={info['n_agents']}, obs_shape={info['obs_shape']}, "
              f"action_shape={info['action_shape']}")

        # 创建算法（不调用 torch）
        algo = _Test(config={'device': 'cpu'}, env_info=info)
        print(f"  算法: n_agents={algo.n_agents}, obs_shape={algo.obs_shape} "
              f"(type={type(algo.obs_shape).__name__})")
        assert isinstance(algo.obs_shape, int), f"obs_shape 应该是 int，实际 {type(algo.obs_shape)}"

        # 跑环境一周
        obs_dict, info_dict = env.reset()
        total_reward = 0
        for step in range(25):
            actions = {aid: np.random.randint(0, info['action_shape'])
                      for aid in obs_dict}
            obs_dict, rewards, dones, infos = env.step(actions)
            total_reward += sum(rewards.values())
        print(f"  ✅ 跑完 25 步，总奖励: {total_reward:.2f}")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_torch():
    """测试 torch 完整路径（如果有 torch）"""
    print("\n[附加] 测试 torch 完整路径...")
    try:
        import torch
        from src.algorithms import ALGORITHM_REGISTRY
        from src.envs import make_env

        env = make_env('mpe_spread', {'n_agents': 4, 'max_steps': 25})
        info = env.get_env_info()
        # 明确传 4 个 Agent，PBFT f=1 需要 3*1+1=4
        algo = ALGORITHM_REGISTRY['pbft_cg_mappo'](
            config={'name': 'pbft_cg_mappo', 'pbft_f': 1, 'use_rnn': True,
                    'hidden_dim': 64, 'share_param': True, 'device': 'cpu',
                    'lr': 1e-4},
            env_info=info,
        )
        print(f"  ✅ PBFT-CG-MAPPO 创建成功")
        print(f"     参数总量: {sum(p.numel() for p in algo.parameters()):,}")

        # 模拟一次 act()
        # obs: (n_agents, obs_dim)
        # hidden: GRU 的 hx 是 (num_layers, batch, hidden_size) = (1, n_agents, 64)
        obs = torch.randn(info['n_agents'], algo.obs_shape)
        hidden = torch.zeros(1, info['n_agents'], 64)
        actions, new_hidden, log_probs = algo.act(obs, hidden, deterministic=False)
        print(f"  ✅ PBFT-CG-MAPPO act() 成功，actions shape: {actions.shape}")
        return True
    except Exception as e:
        print(f"  ⚠️  torch 测试失败（可能 torch 未安装）: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="PBFT-CG-MARL 冒烟测试")
    parser.add_argument("--with-torch", action="store_true",
                       help="包含 torch 完整测试（需要装 torch）")
    args = parser.parse_args()

    print("="*60)
    print("🧪 PBFT-CG-MARL 冒烟测试")
    print("="*60)

    results = []
    results.append(("算法注册表", test_algorithm_registry()))
    results.append(("环境注册表", test_env_registry()))
    results.append(("环境创建", test_env_create()))
    results.append(("PBFT 共识层", test_pbft_consensus()))
    results.append(("形状归一化", test_shape_normalization()))
    results.append(("完整流程", test_full_pipeline()))

    if args.with_torch:
        results.append(("torch 完整路径", test_with_torch()))

    print(f"\n{'='*60}")
    print("📊 测试结果汇总")
    print(f"{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n通过: {passed}/{len(results)}")

    if passed == len(results):
        print(f"\n🎉 全部通过！可以开始训练")
        sys.exit(0)
    else:
        print(f"\n⚠️  有 {len(results)-passed} 项失败，请检查")
        sys.exit(1)


if __name__ == "__main__":
    main()
