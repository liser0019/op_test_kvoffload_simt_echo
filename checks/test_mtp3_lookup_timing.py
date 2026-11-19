"""CPU checks that three device launches count as one Plan sample."""
import csv
import tempfile
import unittest
from pathlib import Path

from profile_joint_mtp import lookup_pipeline_kernels, read_kernels


class LookupTiming(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "kernel_details.csv"
        self.parallel = "joint_mtp3_lookup4_512"
        self.single = "joint_mtp3_full_512"
        self.stages = lookup_pipeline_kernels(self.parallel)
        self.single_kernel, = lookup_pipeline_kernels(self.single)

    def write(self, rows):
        with self.path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Name", "Start Time(us)", "Duration(us)", "Device_id", "Block Num"))
            writer.writerows((name, start, duration, 4, 64) for name, start, duration in rows)

    def test_whole_plan_includes_inter_kernel_gaps(self):
        self.write([(self.stages[0], 100, 12), (self.stages[1], 114, 7),
                    (self.stages[2], 122, 8), (self.single_kernel, 131, 25)])
        order = [self.parallel, self.single]
        parallel, single = read_kernels([self.path], order, 1, order)
        self.assertEqual(parallel["duration_us"], 30)
        self.assertEqual(parallel["busy_us"], 27)
        self.assertEqual(single["duration_us"], 25)

    def test_missing_stage_fails(self):
        self.write([(self.stages[0], 100, 12), (self.stages[2], 112, 8)])
        with self.assertRaises(RuntimeError):
            read_kernels([self.path], [self.parallel], 1, [self.parallel])
