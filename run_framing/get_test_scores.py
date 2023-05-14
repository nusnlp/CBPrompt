""" Compute test scores """
import os
import json
import argparse
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-y', required=True, help="Dir to ground truth test file")
	parser.add_argument('-pred', required=True, help="Dir to test preds")
	args = parser.parse_args()

	# Load ground truth lables
	f = open(os.path.join(args.y, 'test.jsonl'))
	test_y = []
	for line in f.readlines():
		line_dict = json.loads(line)
		test_y.append(line_dict['LBL'])

	print('test_y ', test_y[:10])

	# Load predictions
	f1 = open(os.path.join(args.pred, 'test.json'))
	test_preds = []
	for line in f1.readlines():
		line_dict = json.loads(line)
		test_preds.append(line_dict['label'])

	print('test_preds', test_preds[:10])

	# Get p, r, f, accuracy
	test_acc = accuracy_score(test_y, test_preds)
	print('Overall test acc: ', test_acc)
	# precision_mi, recall_mi, f1score_mi, _ = precision_recall_fscore_support(test_y, test_preds, average='micro')
	precision_mi, recall_mi, f1score_mi, _ = precision_recall_fscore_support(test_y, test_preds, average='micro', labels=["DDI-effect", "DDI-mechanism", "DDI-advise", "DDI-int"])
	print('4 DDI types test precision_mi, recall_mi, f1score_mi', precision_mi, recall_mi, f1score_mi)
	score_report = classification_report(test_y, test_preds)
	print(score_report)
	# Get confusion matrix
	conf_matrix = pd.DataFrame(
	    confusion_matrix(test_y, test_preds, labels=["0", "DDI-effect", "DDI-mechanism", "DDI-advise", "DDI-int"]), 
	    index=['true:0', 'true:eff', 'true:mech', 'true:adv', 'true:int'], 
	    columns=['pred:0', 'pred:eff', 'pred:mech', 'pred:adv', 'pred:int']
	)
	print(conf_matrix)

	# Save test scores
	with open(os.path.join(args.pred, 'test_scores_cp.txt'), 'w') as fout:
		fout.write('Test acc:\t{}\n'.format(test_acc))
		fout.write('4 DDI types Test precision_mi:\t{}\trecall_mi\t{}\tf1score_mi\t{}\n'.format(precision_mi, recall_mi, f1score_mi))
		fout.write(str(score_report)+'\n')
		fout.write(str(conf_matrix)+'\n')



