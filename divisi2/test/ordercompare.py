def order_compare(s1, s2):
    assert len(s1) == len(s2)
    score = 0
    total = 0
    for i in xrange(len(s1)):
        for j in xrange(i+1, len(s1)):
            total += 1
            if s1[i] < s1[j]:
                if s2[i] < s2[j]: score += 1
                elif s2[i] > s2[j]: score -= 1
            elif s1[i] > s1[j]:
                if s2[i] < s2[j]: score -= 1
                elif s2[i] > s2[j]: score += 1
    return (float(score) / total, score, total)

print order_compare([1,2,3,4,5],[1,2,3,4,5])
print order_compare([1,2,3,4,5],[5,4,3,2,1])
print order_compare([1,2,3,4,5],[1,3,5,2,4])
print order_compare([1,2,3,4,5],[3,2,1,4,5])
