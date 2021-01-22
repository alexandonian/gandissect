# Biggan

```bash
python -m netdissect \
    --gan \
    --model "netdissect.biggan.pretrained_biggan(res=128, pretrained='places365', y=215)" \
    --outdir "dissect/livingroom-biggan-places365" \
    --layer blocks.0.0 blocks.1.0 blocks.2.0 blocks.3.0 blocks.4.0  \
    --batch_size 1 \
    --size 1000 \
    --y 215 \
    --z_dim 120
```

## Biggan-Deep

```bash
python -m netdissect \
    --gan \
    --model "netdissect.biggan.pretrained_biggan_deep(res=256, pretrained='places365-challenge', y=215)" \
    --outdir "dissect/livingroom-biggan_deep-places365" \
    --layer blocks.0.0 blocks.1.0 blocks.2.0 blocks.3.0 blocks.4.0 blocks.5.0 \
    --batch_size 1 \
    --size 1000 \
    --y 215 \
    --z_dim 128
```
