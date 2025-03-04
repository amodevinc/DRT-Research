import numpy as np
import pandas as pd
import random
import pickle

# 환경 정의
class Environment:
    def __init__(self, user_history, features):
        self.user_history = user_history
        self.features = features  # 상황별 대안별 feature
        self.x = 0
        

    def get_actual_choice(self):
        """특정 상황에서 실제 사용자가 선택한 대안을 반환"""
        return self.user_history[self.x]

    def step(self, action, actual_choice):
        """에이전트의 행동과 실제 사용자 선택을 비교하여 보상을 반환"""
        if action == actual_choice:
            reward = 1  # 올바른 선택에 대해 보상을 부여
        else:
            reward = -1  # 잘못된 선택에 대해 패널티를 부여
            
        self.move_next()
        done = self.is_done()
        return self.x, reward, done

    def move_next(self):  # 상태 변화
        self.x += 1
        
    def is_done(self):
        return self.x == len(self.user_history) - 1
    
    def get_state(self):
        return self.x
    
    def reset(self):
        self.x = 0
        
        
# 에이전트 정의
class PolicyGradientAgent:
    def __init__(self, alpha, beta, feature_dim,epsilon, gamma=0.99):
        self.alpha = alpha  # 학습률
        self.beta = beta    # Logit 모델의 민감도 파라미터
        self.gamma = gamma  # 할인율
        self.epsilon = epsilon
        self.weights = np.zeros((feature_dim,))  # 가중치 초기화

    def softmax(self, action_values):
        """Logit 모델 기반으로 행동 선택 확률 계산"""
        max_value = np.max(action_values)  # 숫자 안정성을 위한 조치
        exp_values = np.exp(self.beta * (action_values - max_value))  # 오버플로우 방지
        return exp_values / np.sum(exp_values)

    def select_action(self, situation, features):
        """정책에 따라 행동 선택"""
        if np.random.rand()<self.epsilon:
            return np.random.choice(np.arange(len(features[situation]))), [1/len(features[situation])]*len(features[situation])
        else:
            action_values = np.dot(features[situation], self.weights)
            action_probabilities = self.softmax(action_values)
            action = np.argmax(action_probabilities)
            return action, action_probabilities
        
    
    def policy_evaluation(self, history):
        """정책 평가: 리턴 G_t 계산"""
        G = 0
        returns = []
        for _, _, reward, _ in reversed(history):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        return returns

    def policy_improvement(self, history, returns, features):
        """정책 개선: 그라디언트 어센트를 통한 가중치 업데이트"""
        for (action, state, _, action_probabilities), G in zip(history, returns):
            gradient = np.zeros_like(self.weights)
            for a in range(len(action_probabilities)):
                if a == action:
                    gradient += (1 - action_probabilities[a]) * features[state][a]
                else:
                    gradient -= action_probabilities[a] * features[state][a]
            self.weights += self.alpha * G * gradient

    def update_policy(self, history, features):
        """정책 반복: 평가 및 개선 과정"""
        returns = self.policy_evaluation(history)
        self.policy_improvement(history, returns, features)

    def print_weights(self):
        """가중치 출력"""
        print("학습 완료된 가중치:")
        print(self.weights)

# 데이터 콜
def data_call(path):
    # 데이터 불러오기
    user_history_df = pd.read_csv(path+'user_history.csv')
    features_df = pd.read_csv(path+'features.csv')
    return user_history_df, features_df

# 데이터 전처리
def data(id, path, user):
    user_history_df, features_df = data_call(path)
    #이용자별 데이터 추출
    col1=user_history_df.columns[1:]
    col2=features_df.columns[1:]
    
    user.update({id:{"user_history_df":user_history_df.loc[user_history_df["id"]==id,col1],
          "features_df":features_df.loc[features_df["id"]==id,col2]}})
    #예시;
    user_history_df= user[id]["user_history_df"]
    features_df= user[id]["features_df"]

    # user_history를 리스트로 변환
    user_history = user_history_df['choice'].tolist()
    features_df["situation"]=np.repeat(list(range(0,max(user_history_df["situation"])+1)),3)
    col= features_df.columns[2:]
    # features를 numpy array로 변환
    num_situations = features_df['situation'].nunique()
    num_actions = features_df['alternative'].nunique()
    feature_dim = len(features_df.columns) - 2  # situation과 alternative 제외한 나머지 열이 feature

    # features array 구성
    features = np.zeros((num_situations, num_actions, feature_dim))
    for _, row in features_df.iterrows():
        situation = int(row['situation'])
        alternative = int(row['alternative'])
        features[situation, alternative] = row[col]
        
    return user_history, features, num_situations, feature_dim


# 학습 메인 코드
def main(user_history, features,  feature_dim, learned_weights=None):
    alpha = 0.001
    beta = 1.0
    gamma = 0.95
    num_episodes = 100
    epsilon=0.05
    # 환경 및 에이전트 초기화
    env = Environment(user_history, features)
    agent = PolicyGradientAgent(alpha, beta, feature_dim, epsilon,gamma)
    if learned_weights is not None:
        agent.weights = learned_weights
    # 학습 시작
    for k in range(num_episodes):
        done = False
        history = []
        total_reward = 0
        while not done: # done 에피소드 종료 여부 판단
            actual_choice = env.get_actual_choice()
            action, action_probabilities = agent.select_action(env.get_state(), env.features)
            state, reward, done = env.step(action, actual_choice) 
            history.append((action, state, reward, action_probabilities))
            total_reward += reward  
        
        env.reset()
        agent.update_policy(history, features)  # 정책 업데이트

    # 학습 결과 출력
    agent.print_weights()
    
    return agent, env, agent.weights

if __name__ == "__main__":
    total_correct_predictions=0
    user=dict()
    path="C:/Users/TPLAB/Desktop/RL/" # 데이터 경로
    for id in list(range(1,501)):
        user_history, features, num_situations, feature_dim = data( id, path, user)
        # 학습 실행
        if user[id].keys() == "agent":
            trained_agent, trained_env, weights = main(user_history, features, feature_dim, user[id]["agent"].weights)
        else:
            trained_agent, trained_env, weights = main(user_history, features, feature_dim)
            
        num_eval_episodes = 1
        correct_predictions = 0

        for episode in range(num_eval_episodes):
            for situation in range(num_situations):
                actual_choice = user_history[situation]
                action, action_probabilities = trained_agent.select_action(situation, features)
                
                if action == actual_choice:
                    correct_predictions += 1
                    total_correct_predictions +=1
        accuracy = correct_predictions / (num_eval_episodes * num_situations)
        print(f"{id}의 적합도 : {accuracy:.2%}")
        user[id].update({"env":trained_env,"agent":trained_agent, "accuracy":accuracy, "weights":weights})
    accuracy = total_correct_predictions / (num_eval_episodes * num_situations*id)
    print(f"총 예측 정확도: {accuracy:.2%}")