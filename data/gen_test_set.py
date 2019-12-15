
good = open('good_urls.csv', 'r')
bad = open('data.csv', 'r')
final = open('test.csv', 'w')

cnt = 0
num = 0
for line in bad:
    if cnt >= 100:
        break;
    if num % 2:
        final.write(line)
        cnt += 1
    num += 1
for line in good:
    final.write(line)

good.close()
bad.close()
final.close()
