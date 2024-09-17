import numpy as np
from sklearn.linear_model import LogisticRegression


test_data_activations_np = np.load('test_data_activations_70B_quant.npy')
deploy_data_activations_np = np.load('deploy_data_activations_70B_quant.npy')

labels_test = np.zeros(test_data_activations_np.shape[0])
labels_deploy = np.ones(deploy_data_activations_np.shape[0])

data = np.vstack((test_data_activations_np, deploy_data_activations_np))
labels = np.concatenate((labels_test, labels_deploy))

model = LogisticRegression()
model.fit(data, labels)

coefficients = np.squeeze(model.coef_, axis=0).astype(np.float32)
intercept = np.squeeze(model.intercept_, axis=0).astype(np.float32)
np.savez('model_parameters_70B_quant.npz', coefficients=coefficients, intercept=intercept)