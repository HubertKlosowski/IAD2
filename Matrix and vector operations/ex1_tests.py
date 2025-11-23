
import numpy as np

def test_mul_A_v(foo):

	n=10
	m=100
	A=np.random.normal(0,1,size=(n,m))
	v=np.random.normal(0,1,size=(m,1))
	r1=np.round(foo(A,v),3)
	if (r1.shape!=(n,1)):
		print("\x1b[31mNiepoprawny rozmiar wektora wynikowego.\x1b[0m")
	else:
		r2=np.round(np.dot(A,v),3)
		if (r1==r2).all():
			print("\x1b[32mWynik poprawny.\x1b[0m")
		else:
			print("\x1b[31mWynik niepoprawny.\x1b[0m")
	return	
	
def test_mul_A_B(foo):

	n=10
	m=100
	k=200
	A=np.random.normal(0,1,size=(n,m))
	B=np.random.normal(0,1,size=(m,k))
	r1=np.round(foo(A,B),3)
	if (r1.shape!=(n,k)):
		print("\x1b[31mNiepoprawny rozmiar macierzy wynikowej.\x1b[0m")
	else:
		r2=np.round(np.dot(A,B),3)
		if (r1==r2).all():
			print("\x1b[32mWynik poprawny.\x1b[0m")
		else:
			print("\x1b[31mWynik niepoprawny.\x1b[0m")
	return	
	
def test_sum_u_v(u,v,A):

	u1=np.sum(A,axis=1)
	v1=np.sum(A,axis=0)
	if (u1.shape==u.shape and v1.shape==v.shape):
		if ((u1==u).all() and (v1==v).all()):
			print("\x1b[32mWynik poprawny.\x1b[0m")
		else:
			print("\x1b[31mWynik niepoprawny.\x1b[0m")
	else:
		print("\x1b[31mWynik niepoprawny.\x1b[0m")
	return