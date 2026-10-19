/*
 * How a process talks to `tachyon serve`: one Unix socket to connect, then one shared memory for everything else.
 * The messages are in tachyon.xml (SBE); this header is the shared memory's layout and the rules around it. Plain
 * C11: other languages read it as a struct definition.
 *
 * 1. Connect to the daemon's Unix stream socket: $TACHYON_SOCKET, else $XDG_RUNTIME_DIR/tachyon.sock, else
 *    /tmp/tachyon-<uid>.sock. Only processes of the daemon's user are let in (SO_PEERCRED).
 * 2. Receive one Welcome message (tachyon.xml) with the shared memory's file descriptor attached (SCM_RIGHTS), and
 *    map Welcome.size bytes of it, shared, read and write. It starts with a tachyon_control.
 * 3. To ask: write one request message (Chat, Embed, Decide, Load, Unload, List or Shutdown) at request_offset, with its
 *    bytes in the data area as needed, then write the byte TACHYON_DOORBELL_REQUEST to the socket. Write the next
 *    request only after the answer to this one has ended (Done or Error): the daemon reads the request area while
 *    it answers. TACHYON_DOORBELL_CANCEL stops the request being answered (Error, Cancelled).
 * 4. To read the answers: the daemon appends frames to the ring and raises ring_head (release), then adds 1 to
 *    ring_seq and wakes it (futex, FUTEX_WAKE, not private: the memory is shared between processes). The client
 *    keeps ring_tail: while ring_tail < ring_head (acquire), read the frame at ring_offset + ring_tail % ring_bytes
 *    and raise ring_tail (release) past it; when they are equal, read ring_seq, check ring_head again, and wait on
 *    ring_seq with that value (FUTEX_WAIT). The daemon waits for room when the ring is full, so a client that stops
 *    reading stops its own answers only.
 * 5. Close the socket to leave; the daemon drops the shared memory, and stops a request still being answered.
 *
 * A frame is a uint32 length (the message's bytes, header included), then the message, then padding to 8 bytes.
 * Frames start at multiples of 8 and never wrap: where the next frame would not fit before the ring's end, the
 * daemon writes the length TACHYON_FRAME_WRAP and goes on at the ring's start. Everything is little-endian.
 */
#ifndef TACHYON_H
#define TACHYON_H

#include <stdalign.h>
#include <stdint.h>

#define TACHYON_MAGIC 0x59484354u   /* "TCHY" */
#define TACHYON_VERSION 1u          /* of this layout and of tachyon.xml's schema */
#define TACHYON_SCHEMA_ID 7u        /* tachyon.xml's id, in every messageHeader */

#define TACHYON_DOORBELL_REQUEST 1u /* a request is waiting at request_offset */
#define TACHYON_DOORBELL_CANCEL 2u  /* stop the request being answered */

#define TACHYON_FRAME_WRAP 0xFFFFFFFFu

/* The start of the shared memory. The daemon writes the first block once; then each word below has one writer. */
struct tachyon_control {
    uint32_t magic;            /* TACHYON_MAGIC */
    uint32_t version;          /* TACHYON_VERSION */
    uint64_t size;             /* bytes of the whole shared memory */
    uint64_t request_offset;   /* where the client writes a request, from the start */
    uint64_t request_bytes;
    uint64_t ring_offset;      /* the response ring */
    uint64_t ring_bytes;       /* a power of two */
    uint64_t data_offset;      /* the client's bytes: inputs it names, outputs it asks for */
    uint64_t data_bytes;

    alignas(64) uint64_t ring_head;  /* written by the daemon: bytes appended to the ring, ever */
    alignas(64) uint32_t ring_seq;   /* written by the daemon: a futex word, raised after every append */
    alignas(64) uint64_t ring_tail;  /* written by the client: bytes it has read, ever */
};

#endif
