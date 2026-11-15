import pytest
import torch
from helix_serve import HELIXServe, create_helix_serve

class TestHELIXServe:
    def test_init(self):
        m = HELIXServe(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = HELIXServe(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_helix_serve()
        assert isinstance(m, HELIXServe)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
