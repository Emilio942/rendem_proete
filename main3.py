from cv2 import mean
from main import VizdoomEnv
from stable_baselines3 import PPO
import time
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO.load('/home/emilio/Documents/py/work/vizdoom/my/train/train_basic/best_model_350000.zip')   
env = VizdoomEnv(render=True)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

for episode in range(10):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        time.sleep(0.1)
    print(f"Episode {episode+1,rewards} finished")