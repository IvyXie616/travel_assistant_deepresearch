def Solve(n, lens) :
    # write code here
    lens = sorted(lens, key=lambda x: -x)
    while True:
        if not lens:
            break
        N = len(lens)
        if lens[0] >= lens[1] + lens[2]:
            lens = lens[1:N]
        else:
            break
    if len(lens) < 3:
        return 0
        
    cnt = 0
    N = len(lens)
    for i in range(N-2):
        for j in range(i+1, N-1):
            for z in range(j+1, N):
                if lens[i] < lens[j] + lens[z]:
                    cnt += 1
                else: break
    return cnt

res = Solve(5, [2,1,3,1,2])
print(res)
