import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('runs/1lab_symbg/weight0p00000_2023-07-14_22-05-28/results_test.csv')

reg_values = df['dir_penalty']
iou_values = df['mean_iou_pred']

plt.scatter(reg_values, iou_values)
plt.xlabel('Regularization Value')
plt.ylabel('IOU')
plt.grid()
# plt.show()
plt.savefig('../regularization_iou.pdf')
