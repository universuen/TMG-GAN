from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, accuracy_score, \
    confusion_matrix, roc_curve, precision_score

ratio: float = 1
cnt: int = 6

with open(f'data_{ratio}.pkl', 'rb') as f:
    training_samples, training_labels, test_samples, test_labels = pickle.load(f)
print(test_labels)

for i in range(0, len(training_labels)):
    if training_labels[i] != 0:
        training_labels[i] = 1
for j in range(0, len(test_labels)):
    if test_labels[j] != 0:
        test_labels[j] = 1

training_labels = to_categorical(training_labels)
test_labels = to_categorical(test_labels)

x_train = training_samples
y_train = training_labels

x_test = test_samples
y_test = test_labels

Train = np.empty((len(x_train), 13, 6, 1))
print(Train.shape)

for i in range(0, len(x_train)):
    Train[i] = x_train[i].reshape(13, 6, 1)
x_train = Train

Test = np.empty((len(x_test), 13, 6, 1))
for j in range(0, len(x_test)):
    Test[j] = x_test[j].reshape(13, 6, 1)
x_test = Test

n_hidden_1 = 256  # 设定隐藏层
n_classes = 2  # 设定最后的输出层
training_epochs = 200  # 设定整体训练数据共训练多少次
batch_size = 128  # 设定每次提取多少张图片

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(13, 6, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(n_hidden_1, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=training_epochs, batch_size=128, validation_split=0.2, shuffle=True, verbose=1)
model.summary()
score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print(score)

pre = model.predict(x_test)

for i in range(len(pre)):
    max_value = max(pre[i])
    for j in range(len(pre[i])):
        if max_value == pre[i][j]:
            pre[i][j] = 1
        else:
            pre[i][j] = 0

out = []
label = []

for i in range(len(pre)):
    out.append(np.argmax(pre[i]))
    label.append(np.argmax(y_test[i]))
out = np.array(out)
label = np.array(label)

Recall = recall_score(label, out, average='macro')
F1 = f1_score(label, out, labels=None, pos_label=0, average='macro', sample_weight=None)
ACC = accuracy_score(label, out)
CM = confusion_matrix(label, out, labels=None, sample_weight=None)
Pr = precision_score(label, out, labels=None, pos_label=0, average='macro')
ROC = roc_curve(label, pre[:, 1], pos_label=1)
AUC = roc_auc_score(y_test, pre, multi_class='ovo')

print('Recall:', Recall)
print('F1:', F1)
print('ACC:', ACC)
print('Pr:', Pr)
print('AUC:', AUC)
print('CM:', CM)

fpr, tpr, thresholds = roc_curve(label, out, pos_label=1)
fpr = np.array(fpr)
tpr = np.array(tpr)
np.savetxt(f'./0.1/OBP-fpr_{cnt}.txt', fpr)
np.savetxt(f'./0.1/OBP-tpr_{cnt}.txt', tpr)

plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         label='ROC curve (area = %0.2f)' % AUC)  # 假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show()

# CM = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]  # 归一化
# CM = np.around(CM, decimals=2)
plt.figure(figsize=(8, 8))
p = sns.heatmap(CM, annot=True, cmap='Blues', fmt="d")

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
# plt.show()
s = p.get_figure()
s.savefig(f'./0.1/CM-{cnt}.png')
# plt.imsave('0.1-CM.jpg', p)

EMOS = ['0', '1']
NUM_EMO = len(EMOS)
CR = classification_report(y_test, pre, target_names=EMOS, digits=4, labels=list(range(NUM_EMO)), output_dict=True)
print(CR)
df = pd.DataFrame(CR).transpose()
df.to_csv(f"./0.1/CR-{cnt}.csv", index=True)
