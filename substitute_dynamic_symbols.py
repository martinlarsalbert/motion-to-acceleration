import sympy as sp
import sympy.physics.mechanics as me

def substitute_dynamic_symbols(expression):
    dynamic_symbols = me.find_dynamicsymbols(expression)
    derivatives = find_derivatives(dynamic_symbols)
    none_derivatives = dynamic_symbols - derivatives

    # First susbtitute the Derivatives:
    subs = []
    for derivative in list(derivatives):
        name = find_name(derivative)
        symbol = sp.Symbol(name)
        subs.append((derivative, symbol))

    new_expression_derivatives = expression.subs(subs)

    # ...Then substitute the dynamic symbols
    subs = []
    for dynamic_symbol in list(none_derivatives):
        name = find_name(dynamic_symbol=dynamic_symbol)
        symbol = sp.Symbol(name)
        subs.append((dynamic_symbol, symbol))

    new_expression = new_expression_derivatives.subs(subs)

    return new_expression


def find_name(dynamic_symbol):
    if isinstance(dynamic_symbol, sp.Derivative):
        name = find_derivative_name(dynamic_symbol)
    else:
        name = dynamic_symbol.name
    return name

def find_derivatives(dynamic_symbols:set)->set:

    derivatives = []

    for dynamic_symbol in list(dynamic_symbols):
        if isinstance(dynamic_symbol, sp.Derivative):
            derivatives.append(dynamic_symbol)

    derivatives = set(derivatives)
    return derivatives


def find_derivative_name(derivative):

    if not isinstance(derivative, sp.Derivative):
        raise ValueError('%s must be an instance of sympy.Derivative' % derivative)

    order = derivative.args[1][1]
    symbol = derivative.expr

    name = '%s%id' % (symbol.name, order)

    return name


def lambdify(expression):
    new_expression = substitute_dynamic_symbols(expression)
    args = new_expression.free_symbols
    lambda_function = sp.lambdify(args, new_expression, modules='numpy')
    return lambda_function