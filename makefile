network: network.cpp network.h network.i
	swig -c++ -python network.i
	g++ -fPIC -c network.cpp
	g++ -fPIC -c network_wrap.cxx -I/usr/include/python2.7 -I/usr/include/x86_64-linux-gnu/python2.7  -fno-strict-aliasing -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security  -DNDEBUG -g -fwrapv -O2 -Wall

	g++ -shared  network.o network_wrap.o -L/usr/lib/python2.7/config-x86_64-linux-gnu -L/usr/lib -lpthread -ldl  -lutil -lm  -lpython2.7 -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions -o _network.so

test: network.cpp network.h network_test.cpp
	g++ -std=c++11 -Wall -g $? -o $@

clean:
	$(RM) test *.o* *.so* *.pyc *.cxx *.gch network.py 
