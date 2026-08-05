import conf

import properties.P1 as P1
import properties.P2 as P2
import properties.P3 as P3
import properties.P4 as P4
import properties.P5 as P5
import properties.P6 as P6
import properties.P7 as P7
import properties.P8 as P8
import properties.P9 as P9
import properties.P10 as P10
import properties.P123456810 as P123456810
import properties.P123467810 as P123467810
import properties.P123456910 as P123456910
import properties.P12345610 as P12345610

import torch.nn as nn

def soft_loss_proxy(network, N, args):
	L = None
	P = args.property
	if P == 1:
		L = P1.soft_loss
	elif P == 2:
		L = P2.soft_loss
	elif P == 3:
		L = P3.soft_loss
	elif P == 4:
		L = P4.soft_loss
	elif P == 5:
		L = P5.soft_loss
	elif P == 6:
		L = P6.soft_loss
	elif P == 7:
		L = P7.soft_loss
	elif P == 8:
		L = P8.soft_loss
	elif P == 9:
		L = P9.soft_loss
	elif P == 10:
		L = P10.soft_loss
	elif P == 123456810:
		L = P123456810.soft_loss
	elif P == 123467810:
		L = P123467810.soft_loss
	elif P == 123456910:
		L = P123456910.soft_loss
	elif P == 12345610:
		L = P12345610.soft_loss
	else:
		assert False

	return L(network, N, args)

def opportunistic_search_proxy(network, args, N=10000):
	f = None
	P = args.property
	if P == 1:
		f = P1.opportunistic_search
	elif P == 2:
		f = P2.opportunistic_search
	elif P == 3:
		f = P3.opportunistic_search
	elif P == 4:
		f = P4.opportunistic_search
	elif P == 5:
		f = P5.opportunistic_search
	elif P == 6:
		f = P6.opportunistic_search
	elif P == 7:
		f = P7.opportunistic_search
	elif P == 8:
		f = P8.opportunistic_search
	elif P == 9:
		f = P9.opportunistic_search
	elif P == 10:
		f = P10.opportunistic_search
	elif P == 123456810:
		f = P123456810.opportunistic_search
	elif P == 123467810:
		f = P123467810.opportunistic_search
	elif P == 123456910:
		f = P123456910.opportunistic_search
	elif P == 12345610:
		f = P12345610.opportunistic_search
	else:
		assert False

	return f(network, args, N)


def formal_verification_proxy(network, args):
	f = None
	P = args.property
	if P == 1:
		f = P1.formal_verification
	elif P == 2:
		f = P2.formal_verification
	elif P == 3:
		f = P3.formal_verification
	elif P == 4:
		f = P4.formal_verification
	elif P == 5:
		f = P5.formal_verification
	elif P == 6:
		f = P6.formal_verification
	elif P == 7:
		f = P7.formal_verification
	elif P == 8:
		f = P8.formal_verification
	elif P == 9:
		f = P9.formal_verification
	elif P == 10:
		f = P10.formal_verification
	elif P == 123456810:
		f = P123456810.formal_verification
	elif P == 123467810:
		f = P123467810.formal_verification
	elif P == 123456910:
		f = P123456910.formal_verification
	elif P == 12345610:
		f = P12345610.formal_verification
	else:
		assert False

	return f(network, args)

def repair_proxy(CEX, CEY, args):
	f = None
	P = args.property
	if args.property == 1:
		f = P1.repair
	elif P == 2:
		f = P2.repair
	elif P == 3:
		f = P3.repair
	elif P == 4:
		f = P4.repair
	elif P == 5:
		f = P5.repair
	elif P == 6:
		f = P6.repair
	elif P == 7:
		f = P7.repair
	elif P == 8:
		f = P8.repair
	elif P == 9:
		f = P9.repair
	elif P == 10:
		f = P10.repair
	elif P == 123456810:
		f = P123456810.repair
	elif P == 123467810:
		f = P123467810.repair
	elif P == 123456910:
		f = P123456910.repair
	elif P == 12345610:
		f = P12345610.repair
	else:
		assert False

	return f(CEX, CEY, args)

def sampling_proxy(CEX, CEY, args):
	f = None
	P = args.property
	if P == 1:
		f = P1.sampling
	elif P == 2:
		f = P2.sampling
	elif P == 3:
		f = P3.sampling
	elif P == 4:
		f = P4.sampling
	elif P == 5:
		f = P5.sampling
	elif P == 6:
		f = P6.sampling
	elif P == 7:
		f = P7.sampling
	elif P == 8:
		f = P8.sampling
	elif P == 9:
		f = P9.sampling
	elif P == 10:
		f = P10.sampling
	elif P == 123456810:
		f = P123456810.sampling
	elif P == 123467810:
		f = P123467810.sampling
	elif P == 123456910:
		f = P123456910.sampling
	elif P == 12345610:
		f = P12345610.sampling
	else:
		assert False

	return f(CEX, CEY, args)