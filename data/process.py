

test_set = open('good_urls.csv', 'w')

orig_set = open('data.csv', 'r')

cnt = 0
for line in orig_set:
    if cnt >= 500:
        break
    if line.endswith('good\n'):
        test_set.write(line)
        cnt += 1

test_set.close()
orig_set.close()
