# Host ↔ PS UART protocol (draft v0)

Binary-framed messages over **115200 8N1** (Zybo default). PC Python script; PS bare-metal parser.

## Frame

```
| SYNC (2) | CMD (1) | LEN (2 LE) | PAYLOAD | CRC16 (2 LE) |
  0xAA 0x55
```

CRC16-CCITT over CMD+LEN+PAYLOAD.

## Commands (PC → board)

| CMD | Name | Payload |
|-----|------|---------|
| `0x01` | PING | empty → RESP OK |
| `0x10` | LOAD_ACT | 8 × int8 |
| `0x11` | LOAD_WGT_ROW | row_idx u8 + u16 weight_word |
| `0x20` | START_GEMV | empty |
| `0x30` | READ_OUT | row_idx u8 → int32 |

## Responses (board → PC)

| CMD | Name | Payload |
|-----|------|---------|
| `0x80` | OK | optional |
| `0x81` | ERR | code u8 |
| `0x90` | GEMV_RESULT | row u8 + int32 value |

v0 can use **ASCII hex** for debug (`LOAD 0 5\n`) before binary CRC.
