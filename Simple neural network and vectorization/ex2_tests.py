
import numpy as np

def compare(x,y):

    v=np.max(np.abs(x-y))
    if (v>1e-06):
        fd=False
    else:
        fd=True
    return fd

def test_sigmoid(foo):
	
	a=np.array([[1,2,3]])
	r1=1/(1+np.exp(-a))
	r2=foo(a)
	if (compare(r1,r2)):
		print("\x1b[32mCorrect.\x1b[0m")
	else:
		print("\x1b[31mIncorrect.\x1b[0m")
	return
	
def test_init(w,b,X):

	fd=True
	(m,n)=np.shape(X)
	if (np.shape(w)!=(1,n)):
		fd=False
	if (np.shape(b)!=()):
		fd=False
	if (fd==True):
		print("\x1b[32mCorrect.\x1b[0m")
	else:
		print("\x1b[31mIncorrect.\x1b[0m")
	return
	
def test_u(r2,w,b,X):
	
	fd=True
	(m,n)=np.shape(X)
	if (np.shape(r2)!=(1,m)):
		fd=False
	else:
		r1=np.dot(w,np.transpose(X))+b
		if (compare(r1,r2)==False):
			fd=False
	if (fd==True):
		print("\x1b[32mCorrect.\x1b[0m")
	else:
		print("\x1b[31mIncorrect.\x1b[0m")
	return
	
def test_hy(r2,u):
	
	fd=True
	if (np.shape(u)!=np.shape(r2)):
		fd=False
	else:
		r1=1/(1+np.exp(-u))
		if (compare(r1,r2)==False):
			fd=False
	if (fd==True):
		print("\x1b[32mCorrect.\x1b[0m")
	else:
		print("\x1b[31mIncorrect.\x1b[0m")
	return
	
def test_cost(foo):

	m=10
	y=(np.random.randint(0,9,(1,10))+1)/10
	haty=(np.random.randint(0,9,(1,10))+1)/10
	r1=-np.sum(y*np.log(haty)+(1-y)*np.log(1-haty))/m
	r2=foo(y,haty)
	if (np.abs(r1-r2)<1e-06).all():
		print("\x1b[32mCorrect.\x1b[0m")
	else:
		print("\x1b[31mIncorrect.\x1b[0m")
	return
	
def test_training(w,b,y,X):

	(m,n)=X.shape
	u=np.dot(w,np.transpose(X))+b
	haty=1/(1+np.exp(-u))
	y=np.transpose(np.round(y))
	haty=np.round(haty)
	v=0
	for i in range(0,m):
		if (np.abs(y[0,i]-haty[0,i])<1e-3):
			v=v+1
	v=100*v/m
	if (v>=75):
		print("\x1b[32mCorrect.\x1b[0m")
	else:
		print("\x1b[31mClassification accuracy of "+str(v)+"% is too low. Keep learning. The expected value is at least 75%\x1b[0m")
	return