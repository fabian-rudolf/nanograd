import torch
from nanograd.value import Value

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backpropagate()
    x_nanograd, y_nanograd = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    x_tensor, y_tensor = x, y

    # forward pass went well
    assert y_nanograd.of == y_tensor.data.item()
    # backward pass went well
    assert x_nanograd.gradient == x_tensor.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backpropagate()
    a_nanograd, b_nanograd, g_nanograd = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    a_tensor, b_tensor, g_tensor = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(g_nanograd.of - g_tensor.data.item()) < tol
    # backward pass went well
    assert abs(a_nanograd.gradient - a_tensor.grad.item()) < tol
    assert abs(b_nanograd.gradient - b_tensor.grad.item()) < tol


if __name__ == '__main__':
    test_sanity_check()
    test_more_ops()
    print("Success! All tests passed! Nanograd is empirically proven to produce equal results to pytorch.Tensor calculations for the implemented test cases.")
