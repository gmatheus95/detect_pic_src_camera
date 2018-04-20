# Calculate accuracy
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y_validation)]  
accuracy = (sum(map(int, correct)) / float(len(correct)))  
print('accuracy formula= {0}%'.format(accuracy * 100))

# Accuracy score
sklearn.metrics.accuracy_score(y_validation,y_pred, normalize=True):

# Use SPN (wavedec/waverec for noise removal
# Take SNR, mean, variance, skewness and kurtosis from Noise images
# I am using Pywavelets. To denoise I did with CSV2. 

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()

logistic.fit(x_train, y_train)

# VALIDATION

y_pred = logistic.predict(x_validation)
print("Predicted class %s, real class %s" % (y_pred,y_validation))
print ("Probabilities for each class from 1 to 10: %s"
 % logistic.predict_proba(x_validation))
print('Accuracy of logistic regression classifier on validation set: {:.2f}'.format(logistic.score(x_validation, y_validation)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_validation,y_pred)
print(confusion_matrix)
