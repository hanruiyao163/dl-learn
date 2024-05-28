
import threading
import time
 
def test(p):
    time.sleep(1)
    print(p)
 
ts = []
 
for i in range(5):
    # target指定线程要执行的代码，args指定该代码的参数
    t = threading.Thread(target=test, args=[i])
    ts.append(t)
 
for i in ts:
    i.start()
    # i.join()
     
print("it is end !")