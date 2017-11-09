test: network_test.cpp network.cpp network.h
	g++ -std=c++11 -Wall -g $? -o $@
#	g++ -I ~/bin/include -pthread -Wall -g  $? libgtest.a -o $@

clean:
	$(RM) test
