test: network.cpp network.h network_test.cpp
	g++ -std=c++11 -Wall -g $? -o $@
#	g++ -I ~/bin/include -pthread -Wall -g  $? libgtest.a -o $@

clean:
	$(RM) test
