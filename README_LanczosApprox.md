Assume `model` is an instance of class `GTV`. To execute a forward pass with ideal inverse matrix:

```python
outputs = model.forward(inputs) # inverse matrix
```

To execute forward pass with Lanczos approximation instead of inverse matrix:
```python
outputs = model.forward_approx(inputs) # Lanczos approx
```

To modify the Lanczos order, use the below snippet:
```python
model.lanczos_order = desired_order # default is 100, higher order has smaller approx. error and longer runtime
model.support_e1 = torch.zeros(model.lanczos_order, 1).type(model.dtype)
model.support_e1[0] = 1
```

Here is a sample runtime benchmark when `lanczos_order = 20`:
```python
%timeit gtv.gtv1.forward(inputs) # Inverse matrix
# 6.99 s ± 165 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
%timeit gtv.gtv1.forward_approx(inputs) # Lanczos approx
# 4.65 s ± 21.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
