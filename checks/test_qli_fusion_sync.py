"""Source-level reinitialization guard; this does not model real NPU pipelines."""
import unittest
from checks.check_qli_fusion_resources import check_scale_reset


class ScaleReset(unittest.TestCase):
    def test_new_clear_has_cross_pipe_barrier(self):
        check_scale_reset('''void QLIVector<T>::ProcessVec1(const Info& info) {
            Duplicate(kScaleUB_, 0, count);
            PipeBarrier<PIPE_ALL>();
            GetKeyScale(info, kScaleUB_, batch, start, len);
        }''')

    def test_old_ready_flag_cannot_cover_new_clear(self):
        with self.assertRaisesRegex(RuntimeError,'across pipes'):
            check_scale_reset('''void QLIVector<T>::ProcessVec1(const Info& info) {
                Duplicate(kScaleUB_, 0, count);
                PipeBarrier<PIPE_V>();
                WaitFlag<HardEvent::V_MTE2>(OLD_READY);
                GetKeyScale(info, kScaleUB_, batch, start, len);
            }''')


if __name__=='__main__':unittest.main()
