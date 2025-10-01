#ifndef __mytimer_h_
#define __mytimer_h_

#include <sys/time.h>

class mytimer_t {
    private:
	long time_total;
	long time_previous;

	struct timeval time_start;

	struct timeval time_end;

    public:
	mytimer_t() : time_total(0), time_previous(0) {}

	void start() { gettimeofday(&time_start, NULL); }

	void pause() {
	    gettimeofday(&time_end, NULL);
	    time_previous = time_total;
	    time_total += (time_end.tv_sec - time_start.tv_sec) * 1000000 + (time_end.tv_usec - time_start.tv_usec);
	}

	void reset() { time_total = 0; time_previous = 0;}

	double get_current_time() { return time_total * 1e-6; }
	double get_previous_time() { return time_previous * 1e-6; }
};

#endif

