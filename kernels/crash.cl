// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
// Building this program crashes some versions of Intel's OpenCL runtime.
typedef struct {
    uint n_mails;
    uint mails[100];
} mailbox;

__kernel void
run() {
    uint me = get_local_id(0);
    __local mailbox boxes[1][2];
    for (uint ix = 0; ix < 1; ix++) {
        __local mailbox *mb = &boxes[ix][me];
        uint i = 0;
        while (true) {
            uint swap = (i * 2) + 1;
            if (swap >= mb->n_mails) {
                break;
            }
            mb->mails[i] = mb->mails[swap];
            i = swap;
        }
    }
}
