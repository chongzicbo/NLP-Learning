（1）第一步：有监督微调。这里我把它称为generate_model。

（2）第二步：训练一个奖励模型。注意，这个奖励模型和generate_model不是一个模型，我把它称为reward_model。

（3）第三步：在奖励模型的基础上，通过强化学习近端策略优化 （PPO）来优化generate_model，得到最终的模型ppo_generate_model。

作者：西西嘛呦
链接：https://zhuanlan.zhihu.com/p/624565712
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。