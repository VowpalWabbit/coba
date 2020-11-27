import unittest

from coba.data.definitions import Metadata, PartMeta, FullMeta
from coba.data.encoders import StringEncoder, NumericEncoder

class Metadata_Tests(unittest.TestCase):

    def test_init_1(self):
        expected_ignore  = True
        expected_label   = None
        expected_encoder = None

        actual_meta = Metadata(expected_ignore, expected_label, expected_encoder)

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertEqual(actual_meta.encoder, expected_encoder)

    def test_init_2(self):
        expected_ignore  = None
        expected_label   = True
        expected_encoder = None

        actual_meta = Metadata(expected_ignore,expected_label,expected_encoder)

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertEqual(actual_meta.encoder, expected_encoder)

    def test_init_3(self):
        expected_ignore  = None
        expected_label   = None
        expected_encoder = NumericEncoder()

        actual_meta = Metadata(expected_ignore, expected_label, expected_encoder)

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertEqual(actual_meta.encoder, expected_encoder)

    def test_init_4(self):
        expected_ignore  = True
        expected_label   = True
        expected_encoder = NumericEncoder()

        actual_meta = Metadata(expected_ignore, expected_label, expected_encoder)

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertEqual(actual_meta.encoder, expected_encoder)

    def test_clone(self):
        expected_ignore  = True
        expected_label   = True
        expected_encoder = NumericEncoder()

        original_meta = Metadata(expected_ignore, expected_label, expected_encoder)
        clone_meta    = original_meta.clone()

        self.assertEqual(clone_meta.ignore , expected_ignore )
        self.assertEqual(clone_meta.label  , expected_label  ) 
        self.assertEqual(clone_meta.encoder, expected_encoder)

        self.assertNotEqual(clone_meta, original_meta)

    def test_override(self):
        expected_ignore  = False
        expected_label   = False
        expected_encoder = NumericEncoder()

        original_meta = Metadata(True,True,NumericEncoder())
        
        applied_meta = original_meta.override(Metadata(expected_ignore, None, None))
        self.assertEqual(applied_meta.ignore , expected_ignore)

        applied_meta = original_meta.override(Metadata(None, expected_label, None))
        self.assertEqual(applied_meta.label  , expected_label) 

        applied_meta = original_meta.override(Metadata(None, None, expected_encoder))
        self.assertEqual(applied_meta.encoder  , expected_encoder)

class PartMeta_Tests(unittest.TestCase):
    
    def test_from_json_works(self):
        expected_ignore  = None
        expected_label   = False
        expected_encoder = NumericEncoder

        actual_meta = PartMeta.from_json('{ "label":false, "encoding":"numeric" }')

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertIsInstance(actual_meta.encoder, expected_encoder)

    def test_init(self):
        expected_ignore  = None
        expected_label   = None
        expected_encoder = None

        actual_meta = PartMeta()

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  )
        self.assertEqual(actual_meta.encoder, expected_encoder)

class FullMeta_Tests(unittest.TestCase):
    
    def test_from_json_1(self):
        expected_ignore  = True
        expected_label   = False
        expected_encoder = NumericEncoder

        actual_meta = FullMeta.from_json('{ "ignore":true, "label":false, "encoding":"numeric" }')

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertIsInstance(actual_meta.encoder, expected_encoder)

    def test_from_json_2(self):
        with self.assertRaises(Exception) as context:
                FullMeta.from_json('{ "label":false, "encoding":"numeric" }')

    def test_init(self):
        expected_ignore  = False
        expected_label   = False
        expected_encoder = StringEncoder

        actual_meta = FullMeta()

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertIsInstance(actual_meta.encoder, expected_encoder)

if __name__ == '__main__':
    unittest.main()
