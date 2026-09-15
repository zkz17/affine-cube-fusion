import argparse

from tests.test_fuse import test_fuse
from tests.test_branch_split import test_branch_split
from tests.test_absorb_branch import test_absorb_branch
from tests.test_complete_branch import test_extraction_equivalence, test_affine_and_emission, \
    test_direction_key_consistency, test_synthetic_plan
from tests.test_branch_fusion import test_fusion_contract, test_fuse_pool, \
    test_four_case_construction, test_pivot_oracle_alignment
from tests.test_fused_backend import test_backend_equivalence, test_simplify_preserves_semantics, \
    test_differential_stats_larger, test_flag_fallback
from tests.test_global_fusion import test_shift_fusion_contract, test_global_pool_invariant, \
    test_global_backend_equivalence, test_global_backend_stats
from tests.test_closed_form import main as closed_form_suite
from tests.test_premerge import main as premerge_suite
from tests.test_fprm import main as fprm_suite
from tests.test_degree_bound import main as degree_bound_suite
from tests.test_cx_optim import main as cx_optim_suite
from tests.test_gl_search import main as gl_search_suite
from tests.test_fprm_proxy import main as fprm_proxy_suite
from tests.test_macro_m2 import main as macro_m2_suite
from tests.test_cover_generators import main as cover_generators_suite

def main():
    argparser = argparse.ArgumentParser(prog='Test Script')

    argparser.add_argument('-v', '--verbose'           , action='store_true')
    argparser.add_argument('-f', '--fuse_test'         , action='store_true')
    argparser.add_argument('-b', '--branch_split_test' , action='store_true')
    argparser.add_argument('-a', '--absorb_branch_test', action='store_true')
    argparser.add_argument('-c', '--complete_branch_test', action='store_true')
    argparser.add_argument('-g', '--branch_fusion_test', action='store_true')
    argparser.add_argument('-e', '--fused_backend_test', action='store_true')
    argparser.add_argument('-p', '--global_fusion_test', action='store_true')
    argparser.add_argument('-q', '--closed_form_test'  , action='store_true')
    argparser.add_argument('-r', '--premerge_test'     , action='store_true')
    argparser.add_argument('-s', '--fprm_test'         , action='store_true')
    argparser.add_argument('-d', '--degree_bound_test' , action='store_true')
    argparser.add_argument('-x', '--cx_optim_test'     , action='store_true')
    argparser.add_argument('-u', '--gl_search_test'    , action='store_true')
    argparser.add_argument('-w', '--fprm_proxy_test'   , action='store_true')
    argparser.add_argument('-m', '--macro_m2_test'     , action='store_true')
    argparser.add_argument('-y', '--cover_generator_test', action='store_true')

    args = argparser.parse_args()

    if args.fuse_test          : test_fuse(verbose=args.verbose)
    if args.branch_split_test  : test_branch_split(verbose=args.verbose)
    if args.absorb_branch_test : test_absorb_branch(verbose=args.verbose)
    if args.complete_branch_test:
        test_extraction_equivalence()
        test_affine_and_emission()
        test_direction_key_consistency()
        test_synthetic_plan()
    if args.branch_fusion_test:
        test_fusion_contract()
        test_fuse_pool()
        test_four_case_construction()
        test_pivot_oracle_alignment()
    if args.fused_backend_test:
        test_backend_equivalence()
        test_simplify_preserves_semantics()
        test_differential_stats_larger()
        test_flag_fallback()
    if args.global_fusion_test:
        test_shift_fusion_contract()
        test_global_pool_invariant()
        test_global_backend_equivalence()
        test_global_backend_stats()
    if args.closed_form_test:
        closed_form_suite(verbose=args.verbose)
    if args.premerge_test:
        premerge_suite(verbose=args.verbose)
    if args.fprm_test:
        fprm_suite(verbose=args.verbose)
    if args.degree_bound_test:
        degree_bound_suite(verbose=args.verbose)
    if args.cx_optim_test:
        cx_optim_suite(verbose=args.verbose)
    if args.gl_search_test:
        gl_search_suite(verbose=args.verbose)
    if args.fprm_proxy_test:
        fprm_proxy_suite(verbose=args.verbose)
    if args.macro_m2_test:
        macro_m2_suite(verbose=args.verbose)
    if args.cover_generator_test:
        cover_generators_suite(verbose=args.verbose)

if __name__ == '__main__':
    main()
