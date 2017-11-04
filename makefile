test: network_test.cpp network.cpp network.h
	g++ -Wall -g  $? -o $@

clean:
	$(RM) test
