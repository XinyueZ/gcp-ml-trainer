Private and Public Key Creation
====

```
openssl genpkey -algorithm RSA -out private_key.pem -pkeyopt rsa_keygen_bits:2048
```

```
openssl rsa -pubout -in private_key.pem -out public_key.pem
```

```
cat private_key.pem
cat public_key.pem
```
