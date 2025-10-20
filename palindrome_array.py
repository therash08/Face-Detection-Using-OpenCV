
n = int(input())
A = list(map(int, input().split()))

# Check if palindrome
if A == A[::-1]:
    print("YES")
else:
    print("NO")
