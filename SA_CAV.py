import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


test_data_activations_np = np.load('test_data_activations_8B.npy')
deploy_data_activations_np = np.load('deploy_data_activations_8B.npy')

labels_test = np.zeros(test_data_activations_np.shape[0])
labels_deploy = np.ones(deploy_data_activations_np.shape[0])

data = np.vstack((test_data_activations_np, deploy_data_activations_np))
labels = np.concatenate((labels_test, labels_deploy))

model = LogisticRegression()
model.fit(data, labels)

coefficients = model.coef_
intercept = model.intercept_
np.savez('model_parameters.npz', coefficients=coefficients, intercept=intercept)

predictions = model.predict(data)
conf_matrix = confusion_matrix(labels, predictions)
np.save('confusion_matrix.npy', conf_matrix)