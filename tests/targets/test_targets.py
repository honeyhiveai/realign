from agentsim.evaluators.targets import NumericRangeTarget, ContainsTarget

def test_settarget():
    
    settarget = ContainsTarget({'a', 'b', 'c'})
    assert settarget('a')
    assert not settarget('d')
    
    settarget = ContainsTarget(['a', 'b', 'c'])
    assert settarget('a')
    assert not settarget('d')


def test_num_rangetarget():
    numtarget = NumericRangeTarget('[0, 1]')
    assert numtarget(0.5)
    assert not numtarget(1.1)
    assert numtarget(1)
    assert numtarget(0)
    
    numtarget = NumericRangeTarget((0, 1))
    assert numtarget(0.5)
    
    numtarget = NumericRangeTarget([0, 1])
    assert numtarget(0.5)
    
    numtarget = NumericRangeTarget(0.5)
    assert numtarget(0.5)
    
    numtarget = NumericRangeTarget('[0, 1)')
    assert numtarget(0.5)
    assert not numtarget(1)
    
    numtarget = NumericRangeTarget('(0, 1]')
    assert numtarget(0.5)
    assert not numtarget(0)
    
    numtarget = NumericRangeTarget('(0,)')
    assert numtarget(1)
    assert not numtarget(0)
    
    numtarget = NumericRangeTarget('(,1]')
    assert numtarget(0)
    assert numtarget(1)
    assert not numtarget(2)