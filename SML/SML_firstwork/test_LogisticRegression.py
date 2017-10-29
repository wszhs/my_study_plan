import LogisticRegression as lR
# step 1: load data
print "step 1: load data..."
train_x, train_y = lR.load_data('train_data.txt')
test_x, test_y = lR.load_data('test_data.txt')

# step 2: training
print "step 2: training..."
opts = {'alpha': 1, 'maxIter': 1000}
weights = lR.train_log_regression(train_x, train_y, opts)

# step 3: testing
print "step 3: testing..."
accuracy = lR.test_log_regression(test_x, test_y, weights)

# step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
lR.show_log_regression(weights, train_x, train_y)
