
class Value:
    """ 
    stores a single scalar value (a.of) and its gradient (a.gradient)
    
    in natural language notation: applying o on a and b yields c
    in functional notation: o(a, b) = c
    in operator notation: a (o) b = c

    o: operator in (+, *, tanh, higher-order functions)
    a: self
    b: other
    c: output

    """

    def __init__(a, of, _source=(), _operation=''):
        a.of = of
        a.gradient = 0 # by default assume no effect on result
        # internal variables used for autograd graph construction
        a._backpropagate = lambda: None # by default don't define how gradient can be calculated
        a._source_set = set(_source)
        a._op = _operation # the math operation or function that produced this node

    def __add__(a, b):
        b = b if isinstance(b, Value) else Value(b)
        c = Value(a.of + b.of, (a, b), '+')

        def _backpropagate():
            """
            Differential calculus proof:

            By definition, the local gradient of the + operator's children is:
            For h converging towards 0 (lim h -> 0):
            dx / dh = (f(x+h) - f(x)) / h
            By insertion assuming f is + (f:= +):
            da / dh = ((a+b+h) - (a+b)) / h =
            By reordering:
            (a+b+h-a-b) / h =
            (a-a+b-b+h) / h =
            By cancellation:
            h / h = 1

            By applying the chain rule, the global gradient of a is the local gradient times the output gradient (of c).
            By symmetry of the plus operation (a+b=b+a), what applies to a applies to b similarly.
            """
            a.gradient += 1.0 * c.gradient
            b.gradient += 1.0 * c.gradient
        c._backpropagate = _backpropagate

        return c

    def __mul__(a, b):
        """
        Differential calculus proof:

        a's local gradient is defined as:
        For h converging towards 0 (lim h->0)
        dx / dh = (f(x+h) - f(x)) / h
        Given x := a * b; f = * (multiplication operation)
        By insertion:
        da / dh = ((a+h)*b-a*b)/h = 
        By multiplication:
        (a*b+h*b-a*b)/h =
        By reordering:
        (+a*b-a*b+h*b)/h=
        By cancellation:
        (h*b)/h
        By cancellation:
        b
        =>
        da / dh = b

        By symmetry of the multiplication operation (a*b=b*a):
        db / dh = a
        
        By applying the chain rule, 
        a's global gradient is defined by a's local gradient times c's local gradient.
        """
        b = b if isinstance(b, Value) else Value(b)
        c = Value(a.of * b.of, (a, b), '*')

        def _backpropagate():
            a.gradient += b.of * c.gradient
            b.gradient += a.of * c.gradient
        c._backpropagate = _backpropagate

        return c

    def __pow__(a, b):
        assert isinstance(b, (int, float)), NotImplementedError("only supporting int/float powers for now")
        c = Value(a.of**b, (a,), f'**{b}')

        def _backpropagate():
            a.gradient += (b * a.of**(b-1)) * c.gradient
        c._backpropagate = _backpropagate

        return c

    def relu(a):
        c = Value(0 if a.of < 0 else a.of, (a,), 'ReLU')

        def _backpropagate():
            a.gradient += (c.of > 0) * c.gradient
        c._backpropagate = _backpropagate

        return c

    def backpropagate(a):
        """
            Builds the topological order all of the children in the graph
        """
        topo = []
        visited = set()
        def build_topological_order(v):
            if v not in visited:
                visited.add(v)
                for child in v._source_set:
                    build_topological_order(child)
                topo.append(v)
        build_topological_order(a)

        # go one variable at a time and apply the chain rule to get its gradient
        a.gradient = 1
        for v in reversed(topo):
            v._backpropagate()

    def __neg__(a):
        """
        Returns: -a
        """
        return a * -1

    def __radd__(a, b):
        """
        Returns: b + a
        """
        return a + b

    def __sub__(a, b):
        """
        Returns: a - b
        """
        return a + (-b)

    def __rsub__(a, b):
        """
        Returns: b - a
        """
        return b + (-a)

    def __rmul__(a, b):
        """
        Returns: b * a
        """
        return a * b

    def __truediv__(a, b):
        """
        Returns: a / b
        """
        return a * b**-1

    def __rtruediv__(a, b):
        """
        Returns: b / a
        """
        return b * a**-1

    def __repr__(a):
        return f"Value(of={a.of}, gradient={a.gradient})"
