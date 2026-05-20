using AbstractTensors
using AbstractTensors: contractable, register_coordinate_index!, register_basis_index!,
    unregister_index!, index_home_vbundle, validate_indices, validate_contraction,
    is_dual_vbundles, show_registry
using Test

# =========================================
# Helper: reset module-level registries between test sets that need
# a clean slate.  We operate directly on the exported constants so
# each @testset starts from a known state without relying on load order.
# =========================================
function _clear_all_registries!()
    empty!(_MANIFOLDS)
    empty!(_VBUNDLES)
    empty!(_COORDINATE_INDICES); empty!(_BASIS_INDICES)
    empty!(_TENSORS)
    empty!(_METRICS)
    empty!(_BASES)
    empty!(_FRAME_BUNDLES)
end


@testset "AbstractTensors.jl" begin

    # ─────────────────────────────────────────────────────────────────
    @testset "CoordinateIndex — construction and predicates" begin
        _clear_all_registries!()
        @def_manifold TI_M 4 [ti_a, ti_b, ti_c, ti_d] [TIM_B1, TIM_B2, TIM_B3, TIM_B4]

        # @def_manifold binds contravariant CoordinateIndex variables
        @test ti_a isa CoordinateIndex
        @test [ti_a, -ti_b] isa Vector{CoordinateIndex}

        ua = ti_a
        da = -ti_a

        @test ua isa CoordinateIndex
        @test ua.symbol  == :ti_a
        @test ua.vbundle == :tangentTI_M

        @test da.symbol  == :ti_a
        @test da.vbundle == :cotangentTI_M

        @test is_up(ua)
        @test is_down(da)
        @test !is_down(ua)
        @test !is_up(da)
        @test ua.is_up
        @test da.is_down
        @test !ua.is_down
        @test propertynames(ua) == (:symbol, :vbundle, :is_up, :is_down)

        # flip
        @test flip(ua) == da
        @test flip(da) == ua

        # equality and hashing
        @test ua == CoordinateIndex(:ti_a, :tangentTI_M)
        @test ua != da
        @test hash(ua) == hash(CoordinateIndex(:ti_a, :tangentTI_M))

        # symbol-named sibling index
        @test ti_b == CoordinateIndex(:ti_b, :tangentTI_M)
        @test -ti_b == CoordinateIndex(:ti_b, :cotangentTI_M)

        # show does not throw
        @test (repr(ua); true)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "AbstractIndex — is_dual_vbundles / contractable" begin
        _clear_all_registries!()
        @def_manifold TI2_M 4 [ti2_a, ti2_b, ti2_c, ti2_d] [TI2M_B1, TI2M_B2, TI2M_B3, TI2M_B4]

        ua = ti2_a
        da = -ti2_a
        ub = ti2_b

        @test ua.symbol == da.symbol
        @test ua.symbol != ub.symbol

        @test is_dual_vbundles(:tangentTI2_M, :cotangentTI2_M)
        @test is_dual_vbundles(:cotangentTI2_M, :tangentTI2_M)
        @test !is_dual_vbundles(:tangentTI2_M, :tangentTI2_M)

        @test contractable(ua, da)
        @test contractable(da, ua)
        @test !contractable(ua, ub)
        @test !contractable(ua, ti2_b)

        # cannot contract coordinate with basis index (even same symbol name)
        @test TI2M_B1 isa BasisIndex
        @test !contractable(ua, -TI2M_B1)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "BasisIndex — construction and registry" begin
        _clear_all_registries!()
        @def_manifold BI_M 4 [bi_a1, bi_a2, bi_a3, bi_a4] [BI_A1, BI_A2, BI_A3, BI_A4]

        @test BI_A1 isa BasisIndex
        @test BI_A1.vbundle == :tangentBI_M
        @test (-BI_A1).vbundle == :cotangentBI_M
        @test haskey(_BASIS_INDICES, :BI_A1)
        @test !haskey(_COORDINATE_INDICES, :BI_A1)
        @test haskey(_COORDINATE_INDICES, :bi_a1)

        @test tangentBI_M.basis_indices isa Vector{BasisIndex}
        @test length(tangentBI_M.basis_indices) == 4
        @test tangentBI_M.basis_indices[1].symbol == :BI_A1
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "Index registry — register / unregister / validate" begin
        _clear_all_registries!()

        register_coordinate_index!(:reg_x, :tangentREG)
        @test haskey(_COORDINATE_INDICES, :reg_x)
        @test index_home_vbundle(:reg_x) == :tangentREG

        # idempotent re-registration to same bundle
        @test (register_coordinate_index!(:reg_x, :tangentREG); true)

        # conflict → error
        @test_throws ErrorException register_coordinate_index!(:reg_x, :tangentOTHER)

        register_basis_index!(:reg_b, :tangentREG)
        @test haskey(_BASIS_INDICES, :reg_b)

        # cross-registry collision
        @test_throws ErrorException register_basis_index!(:reg_x, :tangentREG)

        unregister_index!(:reg_x)
        unregister_index!(:reg_b)
        @test !haskey(_COORDINATE_INDICES, :reg_x)
        @test !haskey(_BASIS_INDICES, :reg_b)

        # unregistered access → error
        @test_throws ErrorException index_home_vbundle(:not_registered)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@add_indices" begin
        _clear_all_registries!()
        @def_manifold AI_M 4 [ai_a1, ai_a2, ai_a3, ai_a4] [AIM_B1, AIM_B2, AIM_B3, AIM_B4]

        @add_indices AI_M ai_a5 ai_a6
        @test haskey(_COORDINATE_INDICES, :ai_a5)
        @test haskey(_COORDINATE_INDICES, :ai_a6)
        @test ai_a5.vbundle == :tangentAI_M
        @test ai_a5 == CoordinateIndex(:ai_a5, :tangentAI_M)
        @test -ai_a6 == CoordinateIndex(:ai_a6, :cotangentAI_M)
        @test length(tangentAI_M.coordinate_indices) == 6
        @test length(cotangentAI_M.coordinate_indices) == 6

        # adding to unregistered manifold → error
        @test_throws ErrorException @eval @add_indices FAKE_M ai_z1
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_vbundle — dual field and flip" begin
        _clear_all_registries!()
        @def_manifold VB_M 4 [vb_a1, vb_a2, vb_a3, vb_a4] [VBM_B1, VBM_B2, VBM_B3, VBM_B4]
        @def_vbundle VB_E VB_M 3 [VB_A1, VB_A2, VB_A3]

        @test VB_E.dual == :dualVB_E
        @test dualVB_E.dual == :VB_E
        @test VB_A1 isa BasisIndex
        @test VB_A1.vbundle == :VB_E
        @test (-VB_A1).vbundle == :dualVB_E
        @test is_dual_vbundles(:VB_E, :dualVB_E)
        @test is_dual_vbundles(:dualVB_E, :VB_E)
        @test !is_dual_vbundles(:VB_E, :cotangentVB_M)
        @test [VB_A1, -VB_A2, -VB_A3] isa Vector{BasisIndex}
        @test isempty(VB_E.coordinate_indices)
        @test length(VB_E.basis_indices) == 3
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "Manifold and VBundle structs" begin
        _clear_all_registries!()
        @def_manifold MAN_M 4 [mn_a1, mn_a2, mn_a3, mn_a4] [MANM_B1, MANM_B2, MANM_B3, MANM_B4]

        @test MAN_M isa Manifold
        @test !(42 isa Manifold)
        @test MAN_M.name == :MAN_M
        @test MAN_M.dim  == 4
        @test MAN_M.tangent_bundle   == :tangentMAN_M
        @test MAN_M.cotangent_bundle == :cotangentMAN_M
        @test :tangentMAN_M   in MAN_M.vbundles
        @test :cotangentMAN_M in MAN_M.vbundles

        # VBundle accessors
        tb = tangentMAN_M
        @test tb isa VBundle
        @test !tb.isdual
        @test tb.isdual   == false
        @test tb.dim      == 4
        @test tb.manifold == :MAN_M

        cb = cotangentMAN_M
        @test cb.isdual
        @test cb.isdual == true

        # dot access on Manifold / VBundle
        @test MAN_M.dim             == 4
        @test MAN_M.tangent_bundle   == :tangentMAN_M
        @test MAN_M.cotangent_bundle == :cotangentMAN_M
        @test :tangentMAN_M in MAN_M.vbundles
        @test tb.coordinate_indices == _VBUNDLES[:tangentMAN_M].coordinate_indices
        @test length(tb.coordinate_indices) == 4
        @test length(tb.basis_indices) == 4
        @test :coordinate_indices in propertynames(tangentMAN_M)

        # VBundle.dual field
        @test tangentMAN_M.dual   == :cotangentMAN_M
        @test cotangentMAN_M.dual == :tangentMAN_M
        @test (-mn_a1).vbundle == :cotangentMAN_M
        @test flip(mn_a1).vbundle == :cotangentMAN_M

        # show does not throw
        @test (repr(MAN_M); true)
        @test (repr(tb);    true)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_manifold / @undef_manifold" begin
        _clear_all_registries!()
        @def_manifold DM_M 3 [dm_a, dm_b, dm_c] [Dm_A1, Dm_A2, Dm_A3]

        @test haskey(_MANIFOLDS, :DM_M)
        @test haskey(_VBUNDLES,  :tangentDM_M)
        @test haskey(_VBUNDLES,  :cotangentDM_M)
        @test haskey(_COORDINATE_INDICES, :dm_a)
        @test :DM_M in keys(_MANIFOLDS)

        # redefine issues warning but succeeds
        @test_warn r"already defined" @def_manifold DM_M 3 [dm_a, dm_b, dm_c] [Dm_A1, Dm_A2, Dm_A3]

        # undefine
        @undef_manifold DM_M
        @test !haskey(_MANIFOLDS, :DM_M)
        @test !haskey(_VBUNDLES,  :tangentDM_M)
        @test !haskey(_COORDINATE_INDICES, :dm_a)
        @test !haskey(_BASIS_INDICES, :Dm_A1)

        # undef of non-existent → error
        @test_throws ErrorException @eval @undef_manifold GHOST_M

        # wrong arg type
        @test_throws ErrorException macroexpand(@__MODULE__, :(@def_manifold 123 4 [x] [X1]))
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "validate_indices" begin
        _clear_all_registries!()
        @def_manifold VI_M 4 [vi_a, vi_b, vi_c, vi_d] [VIM_B1, VIM_B2, VIM_B3, VIM_B4]

        # valid
        @test (validate_indices([:vi_a, :vi_b], :tangentVI_M); true)

        # unregistered index
        @test_throws ErrorException validate_indices([:vi_a, :NOT_REG], :tangentVI_M)

        # wrong bundle
        @test_throws ErrorException validate_indices([:vi_a], :tangentOTHER)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "validate_contraction" begin
        _clear_all_registries!()
        @def_manifold VC_M 4 [vc_a, vc_b, vc_c, vc_d] [VCM_B1, VCM_B2, VCM_B3, VCM_B4]

        ua = vc_a; da = -vc_a; ub = vc_b

        @test (validate_contraction(ua, da); true)
        @test_throws ErrorException validate_contraction(ua, ub)   # different symbol
        @test_throws ErrorException validate_contraction(ua, ua)   # same variance
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "SignedPerm — construction and validation" begin
        id2  = identity_perm(2)
        swap = SignedPerm([2, 1], Int8(1))
        neg  = SignedPerm([2, 1], Int8(-1))

        @test id2.images == [1, 2]
        @test id2.sign   == Int8(1)
        @test is_valid_perm(id2)
        @test is_valid_perm(swap)

        # bad sign
        @test !is_valid_perm(SignedPerm([1, 2], Int8(2)))
        # repeated image
        @test !is_valid_perm(SignedPerm([1, 1], Int8(1)))

        @test swap == SignedPerm([2, 1], Int8(1))
        @test swap != neg
        @test hash(swap) == hash(SignedPerm([2, 1], Int8(1)))

        @test (repr(swap); true)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "SignedPerm — compose, inv, apply" begin
        id2  = identity_perm(2)
        swap = SignedPerm([2, 1], Int8(1))
        neg  = SignedPerm([2, 1], Int8(-1))

        @test compose(swap, swap) == id2
        @test compose(neg,  neg)  == id2
        @test compose(neg, swap).sign == Int8(-1)

        @test inv(swap) == swap
        @test inv(neg)  == neg
        @test compose(swap, inv(swap)) == id2

        @test apply(swap, [:a, :b]) == [:b, :a]
        @test apply(id2,  [:a, :b]) == [:a, :b]

        # degree mismatch
        @test_throws ErrorException compose(identity_perm(2), identity_perm(3))
        @test_throws ErrorException apply(swap, [:a])
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "SlotSymmetry — group closure and orders" begin
        @test length(no_symmetry(3).group_elements) == 1

        sym2  = symmetric(2)
        @test length(sym2.group_elements) == 2
        @test is_trivial_symmetry(no_symmetry(4))
        @test !is_trivial_symmetry(sym2)

        @test length(symmetric(3).group_elements)     == 6
        @test length(symmetric(4).group_elements)     == 24
        @test length(antisymmetric(2).group_elements) == 2
        @test length(antisymmetric(3).group_elements) == 6
        @test length(riemann_symmetry().group_elements) == 8

        # symmetric_on / antisymmetric_on
        sym_on = symmetric_on(4, [1, 2])
        @test length(sym_on.group_elements) == 2

        asym_on = antisymmetric_on(4, [3, 4])
        @test length(asym_on.group_elements) == 2

        # empty positions → trivial
        @test is_trivial_symmetry(symmetric_on(4, Int[]))
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "SlotSymmetry — is_in_symmetry" begin
        sym = symmetric(2)
        id2 = identity_perm(2)
        sw  = SignedPerm([2, 1], Int8(1))
        neg = SignedPerm([2, 1], Int8(-1))

        @test is_in_symmetry(id2, sym)
        @test is_in_symmetry(sw,  sym)
        @test !is_in_symmetry(neg, sym)   # wrong sign for symmetric

        asym = antisymmetric(2)
        @test is_in_symmetry(id2, asym)   # identity always in
        @test !is_in_symmetry(sw,  asym)
        @test is_in_symmetry(neg,  asym)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "SlotSymmetry — constructor validation" begin
        # wrong degree for generator
        @test_throws ErrorException SlotSymmetry(2, [SignedPerm([1,2,3], Int8(1))])
        # invalid perm
        @test_throws ErrorException SlotSymmetry(2, [SignedPerm([1,1], Int8(1))])
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "canonical_rep" begin
        _clear_all_registries!()
        @def_manifold CR_M 4 [cr_a, cr_b, cr_c, cr_d] [CRM_B1, CRM_B2, CRM_B3, CRM_B4]

        a = CoordinateIndex(:cr_a, :tangentCR_M)
        b = CoordinateIndex(:cr_b, :tangentCR_M)
        c = CoordinateIndex(:cr_c, :tangentCR_M)

        asym2 = antisymmetric(2)

        # [a, b] is already canonical; sign = +1
        can, sgn = canonical_rep([a, b], asym2)
        @test can == [a, b]
        @test sgn == Int8(1)

        # [b, a] → canonical [a, b] with sign -1
        can2, sgn2 = canonical_rep([b, a], asym2)
        @test can2 == [a, b]
        @test sgn2 == Int8(-1)

        # symmetric: both orders same sign
        sym2 = symmetric(2)
        can3, sgn3 = canonical_rep([b, a], sym2)
        @test can3 == [a, b]
        @test sgn3 == Int8(1)

        # trivial symmetry: always returns input unchanged
        none = no_symmetry(3)
        can4, sgn4 = canonical_rep([c, b, a], none)
        @test can4 == [c, b, a]
        @test sgn4 == Int8(1)

        # degree mismatch
        @test_throws ErrorException canonical_rep([a, b, c], asym2)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "CoordinateIndex isless (used by canonical_rep)" begin
        _clear_all_registries!()
        @def_manifold IL_M 4 [il_a, il_b, il_c, il_d] [ILM_B1, ILM_B2, ILM_B3, ILM_B4]

        a_up = CoordinateIndex(:il_a, :tangentIL_M)
        b_up = CoordinateIndex(:il_b, :tangentIL_M)
        a_dn = CoordinateIndex(:il_a, :cotangentIL_M)

        @test a_up < b_up     # :il_a < :il_b lexicographically
        @test !(b_up < a_up)
        # cotangent < tangent  ("cotangent..." < "tangent..." lexicographically)
        @test a_dn < a_up
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "Tensor struct and accessors" begin
        _clear_all_registries!()
        @def_manifold TS_M 4 [ts_a, ts_b, ts_c, ts_d] [TSM_B1, TSM_B2, TSM_B3, TSM_B4]
        @def_metric ts_g TS_M

        @def_tensor TS_T [cotangentTS_M, cotangentTS_M]

        @test TS_T isa Tensor
        @test is_tensor(TS_T)
        @test !is_tensor(42)

        @test TS_T.manifold    == :TS_M
        @test TS_T.rank        == 2
        @test TS_T.print_as    == :TS_T
        @test TS_T.is_traceless == false
        @test TS_T.metric      == :ts_g
        @test TS_T.slots == [:cotangentTS_M, :cotangentTS_M]

        # propertynames includes :metric and :rank
        pn = propertynames(TS_T)
        @test :metric in pn
        @test :rank   in pn
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_tensor — slot vbundles" begin
        _clear_all_registries!()
        @def_manifold SV_M 4 [sv_a, sv_b, sv_c, sv_d] [SVM_B1, SVM_B2, SVM_B3, SVM_B4]
        @def_metric sv_g SV_M

        # all covariant
        @def_tensor SV_cov [cotangentSV_M, cotangentSV_M]
        @test SV_cov.slots == [:cotangentSV_M, :cotangentSV_M]

        # all contravariant
        @def_tensor SV_con [tangentSV_M, tangentSV_M]
        @test SV_con.slots == [:tangentSV_M, :tangentSV_M]

        # mixed
        @def_tensor SV_mix [tangentSV_M, cotangentSV_M]
        @test SV_mix.slots == [:tangentSV_M, :cotangentSV_M]
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_tensor — symmetries" begin
        _clear_all_registries!()
        @def_manifold TSYM_M 4 [tsym_a, tsym_b, tsym_c, tsym_d] [TSYMM_B1, TSYMM_B2, TSYMM_B3, TSYMM_B4]
        @def_metric tsym_g TSYM_M

        # default: [no_symmetry(n)]
        @def_tensor TSYM_none [cotangentTSYM_M, cotangentTSYM_M]
        @test length(TSYM_none.symmetries) == 1
        @test is_trivial_symmetry(TSYM_none.symmetries[1])

        # single SlotSymmetry (not wrapped in vector)
        @def_tensor TSYM_s [cotangentTSYM_M, cotangentTSYM_M] symmetries=[symmetric(2)]
        @test length(TSYM_s.symmetries[1].group_elements) == 2

        # vector form
        @def_tensor TSYM_v [cotangentTSYM_M, cotangentTSYM_M] symmetries=[antisymmetric(2)]
        @test !is_trivial_symmetry(TSYM_v.symmetries[1])

        # Riemann symmetry (rank-4)
        @def_tensor TSYM_R [cotangentTSYM_M, cotangentTSYM_M, cotangentTSYM_M, cotangentTSYM_M] symmetries=[riemann_symmetry()]
        @test length(TSYM_R.symmetries[1].group_elements) == 8

        # traceless flag
        @def_tensor TSYM_W [cotangentTSYM_M, cotangentTSYM_M, cotangentTSYM_M, cotangentTSYM_M] symmetries=[riemann_symmetry()] traceless=true
        @test TSYM_W.is_traceless

        # symmetry degree mismatch → error
        @test_throws ErrorException @eval begin
            @def_manifold ERR_SYM_M 4 [err_s_a, err_s_b, err_s_c, err_s_d] [ERRSYMM_B1, ERRSYMM_B2, ERRSYMM_B3, ERRSYMM_B4]
            @def_tensor ERR_SYM_T [cotangentERR_SYM_M, cotangentERR_SYM_M] symmetries=[symmetric(3)]
        end
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_tensor — print_as" begin
        _clear_all_registries!()
        @def_manifold PA_M 4 [pa_a, pa_b, pa_c, pa_d] [PAM_B1, PAM_B2, PAM_B3, PAM_B4]
        @def_metric pa_g PA_M

        @def_tensor PA_T [cotangentPA_M, cotangentPA_M] print_as=:Riemann
        @test PA_T.print_as == :Riemann
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_tensor — metric resolution: one metric (silent)" begin
        _clear_all_registries!()
        @def_manifold MR1_M 4 [mr1_a, mr1_b, mr1_c, mr1_d] [MR1M_B1, MR1M_B2, MR1M_B3, MR1M_B4]
        @def_metric mr1_g MR1_M

        @def_tensor MR1_T [cotangentMR1_M, cotangentMR1_M]   # no metric= given
        @test MR1_T.metric == :mr1_g
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_tensor — metric resolution: no metric (warning + nothing)" begin
        _clear_all_registries!()
        @def_manifold MR0_M 4 [mr0_a, mr0_b, mr0_c, mr0_d] [MR0M_B1, MR0M_B2, MR0M_B3, MR0M_B4]

        @test_warn r"No metric" @def_tensor MR0_T [cotangentMR0_M, cotangentMR0_M]
        @test _TENSORS[:MR0_T].metric === nothing
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_tensor — metric resolution: multiple metrics (warning + first)" begin
        _clear_all_registries!()
        @def_manifold MR2_M 4 [mr2_a, mr2_b, mr2_c, mr2_d] [MR2M_B1, MR2M_B2, MR2M_B3, MR2M_B4]
        @def_metric mr2_g MR2_M
        @def_metric mr2_h MR2_M

        @test_warn r"Multiple metrics" @def_tensor MR2_T [cotangentMR2_M, cotangentMR2_M]
        # assigned metric is the first sorted name (:mr2_g < :mr2_h lexicographically)
        @test _TENSORS[:MR2_T].metric in (:mr2_g, :mr2_h)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_tensor — metric= explicit (bare and quoted)" begin
        _clear_all_registries!()
        @def_manifold ME_M 4 [me_a, me_b, me_c, me_d] [MEM_B1, MEM_B2, MEM_B3, MEM_B4]
        @def_metric me_g ME_M
        @def_metric me_h ME_M

        @def_tensor ME_T1 [cotangentME_M, cotangentME_M] metric=me_g
        @test ME_T1.metric == :me_g

        @def_tensor ME_T2 [cotangentME_M, cotangentME_M] metric=:me_h
        @test ME_T2.metric == :me_h
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_tensor — metric= wrong manifold → error" begin
        _clear_all_registries!()
        @def_manifold WM_A 4 [wm_a1, wm_a2, wm_a3, wm_a4] [WMA_B1, WMA_B2, WMA_B3, WMA_B4]
        @def_manifold WM_B 4 [wm_b1, wm_b2, wm_b3, wm_b4] [WMB_B1, WMB_B2, WMB_B3, WMB_B4]
        @def_metric wm_gA WM_A

        @test_throws ErrorException @eval begin
            using AbstractTensors
            @def_tensor WM_T [cotangentWM_B, cotangentWM_B] metric=wm_gA
        end
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_tensor — metric= unregistered symbol → error" begin
        _clear_all_registries!()
        @def_manifold UR_M 4 [ur_a, ur_b, ur_c, ur_d] [URM_B1, URM_B2, URM_B3, URM_B4]

        @test_throws ErrorException @eval begin
            using AbstractTensors
            @def_tensor UR_T [cotangentUR_M, cotangentUR_M] metric=:not_a_metric
        end
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_tensor / @undef_tensor registry bookkeeping" begin
        _clear_all_registries!()
        @def_manifold REG_M 4 [reg_a, reg_b, reg_c, reg_d] [REGM_B1, REGM_B2, REGM_B3, REGM_B4]

        @test_warn r"No metric" @def_tensor REG_T [cotangentREG_M, cotangentREG_M]
        @test haskey(_TENSORS, :REG_T)
        @test :REG_T in list_tensors()

        # redefine issues warning
        @test_warn r"already defined" @def_tensor REG_T [cotangentREG_M, cotangentREG_M]

        @undef_tensor REG_T
        @test !haskey(_TENSORS, :REG_T)
        @test !(:REG_T in list_tensors())

        # undef of non-existent → error
        @test_throws ErrorException @eval @undef_tensor GHOST_TENSOR
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@undef_metric removes from _METRICS and _TENSORS" begin
        _clear_all_registries!()
        @def_manifold UM_M 4 [um_a, um_b, um_c, um_d] [UMM_B1, UMM_B2, UMM_B3, UMM_B4]
        @def_metric um_g UM_M

        @test haskey(_METRICS, :um_g)
        @test_throws ErrorException @undef_tensor um_g
        @undef_metric um_g
        @test !haskey(_TENSORS, :um_g)
        @test !haskey(_METRICS, :um_g)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "Tensor show methods do not throw" begin
        _clear_all_registries!()
        @def_manifold SH_M 4 [sh_a, sh_b, sh_c, sh_d] [SHM_B1, SHM_B2, SHM_B3, SHM_B4]
        @def_metric sh_g SH_M
        @def_tensor SH_T [cotangentSH_M, cotangentSH_M]

        @test (repr(SH_T); true)
        @test (repr("text/html", SH_T); true)
        @test (tensor_info(SH_T); true)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_metric — basic registration" begin
        _clear_all_registries!()
        @def_manifold DFM_M 4 [dfm_a, dfm_b, dfm_c, dfm_d] [DFMM_B1, DFMM_B2, DFMM_B3, DFMM_B4]
        @def_metric dfm_g DFM_M

        @test haskey(_TENSORS, :dfm_g)
        @test haskey(_METRICS, :dfm_g)
        @test _METRICS[:dfm_g] == :DFM_M
        @test is_metric(dfm_g)
        @test :dfm_g in list_metrics()

        # struct properties
        @test dfm_g.rank        == 2
        @test dfm_g.is_traceless == false
        @test dfm_g.metric       == :dfm_g   # self-referential
        @test dfm_g.slots == [:cotangentDFM_M, :cotangentDFM_M]
        @test length(dfm_g.symmetries) == 1
        @test length(dfm_g.symmetries[1].group_elements) == 2   # symmetric(2)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_metric — unregistered manifold → error" begin
        _clear_all_registries!()
        @test_throws ErrorException @eval @def_metric bad_g BAD_MANIFOLD
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_metric — non-metric is_metric = false" begin
        _clear_all_registries!()
        @def_manifold NM_M 4 [nm_a, nm_b, nm_c, nm_d] [NMM_B1, NMM_B2, NMM_B3, NMM_B4]
        @def_metric nm_g NM_M

        @def_tensor nm_T [cotangentNM_M, cotangentNM_M] metric=:nm_g

        # nm_T is a plain tensor; nm_g is the metric
        @test !is_metric(nm_T)
        @test is_metric(nm_g)
        @test !is_metric(42)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@undef_metric" begin
        _clear_all_registries!()
        @def_manifold UMT_M 4 [umt_a, umt_b, umt_c, umt_d] [UMTM_B1, UMTM_B2, UMTM_B3, UMTM_B4]
        @def_metric umt_g UMT_M

        @test haskey(_METRICS, :umt_g)
        @undef_metric umt_g
        @test !haskey(_METRICS, :umt_g)
        @test !haskey(_TENSORS, :umt_g)

        # undef of non-existent → error
        @test_throws ErrorException @eval @undef_metric GHOST_METRIC
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "metrics_of_manifold" begin
        _clear_all_registries!()
        @def_manifold MON_M 4 [mon_a, mon_b, mon_c, mon_d] [MONM_B1, MONM_B2, MONM_B3, MONM_B4]

        @test metrics_of_manifold(:MON_M) == Symbol[]

        @def_metric mon_g MON_M
        @test metrics_of_manifold(:MON_M) == [:mon_g]

        @def_metric mon_h MON_M
        result = metrics_of_manifold(:MON_M)
        @test length(result) == 2
        @test :mon_g in result
        @test :mon_h in result
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "metric_info / show_metrics do not throw" begin
        _clear_all_registries!()
        @def_manifold MI_M 4 [mi_a, mi_b, mi_c, mi_d] [MIM_B1, MIM_B2, MIM_B3, MIM_B4]
        @def_metric mi_g MI_M

        @test (metric_info(mi_g); true)
        @test (show_metrics(); true)
        @test (list_metrics(); true)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "show_registry does not throw" begin
        _clear_all_registries!()
        @def_manifold LM_M 4 [lm_a, lm_b, lm_c, lm_d] [LMM_B1, LMM_B2, LMM_B3, LMM_B4]

        @test :LM_M in keys(_MANIFOLDS)
        @test (show_registry(); true)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "Full workflow: manifold → metric → tensors → symmetry" begin
        _clear_all_registries!()
        @def_manifold FW_M 4 [fw_a, fw_b, fw_c, fw_d] [FWM_B1, FWM_B2, FWM_B3, FWM_B4]
        @def_metric fw_g FW_M

        @def_tensor fw_R [cotangentFW_M, cotangentFW_M, cotangentFW_M, cotangentFW_M] symmetries=[riemann_symmetry()]
        @def_tensor fw_W [cotangentFW_M, cotangentFW_M, cotangentFW_M, cotangentFW_M] symmetries=[riemann_symmetry()] traceless=true print_as=:Weyl

        @test fw_R.metric     == :fw_g
        @test fw_W.print_as   == :Weyl
        @test fw_W.is_traceless

        # canonical_rep over Riemann symmetry
        a = CoordinateIndex(:fw_a, :cotangentFW_M)
        b = CoordinateIndex(:fw_b, :cotangentFW_M)
        c = CoordinateIndex(:fw_c, :cotangentFW_M)
        d = CoordinateIndex(:fw_d, :cotangentFW_M)

        # R[b,a,c,d] = -R[a,b,c,d]
        can, sgn = canonical_rep([b, a, c, d], fw_R.symmetries[1])
        @test sgn == Int8(-1)
        @test can[1] == a   # canonicalized: first two slots sorted

        # R[a,b,d,c] = -R[a,b,c,d]
        can2, sgn2 = canonical_rep([a, b, d, c], fw_R.symmetries[1])
        @test sgn2 == Int8(-1)

        # R[c,d,a,b] = +R[a,b,c,d]  (pair swap)
        can3, sgn3 = canonical_rep([c, d, a, b], fw_R.symmetries[1])
        @test sgn3 == Int8(1)

        # Mixed-rank tensor
        @def_tensor fw_mixed [tangentFW_M, cotangentFW_M]
        @test fw_mixed.slots == [:tangentFW_M, :cotangentFW_M]
        @test fw_mixed.metric == :fw_g
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "Basis and _BASES registry" begin
        _clear_all_registries!()
        @def_manifold BA_M 4 [ba_a, ba_b, ba_c, ba_d] [BAM_B1, BAM_B2, BAM_B3, BAM_B4]

        # @def_manifold registers coordinate frames with defaults ∂/dx
        @test haskey(_BASES, (:tangentBA_M,   :coordinate))
        @test haskey(_BASES, (:cotangentBA_M, :coordinate))
        # ... and moving frames with defaults e/θ
        @test haskey(_BASES, (:tangentBA_M,   :moving))
        @test haskey(_BASES, (:cotangentBA_M, :moving))

        tb_basis  = _BASES[(:tangentBA_M,   :coordinate)]
        ctb_basis = _BASES[(:cotangentBA_M, :coordinate)]

        @test tb_basis  isa Basis
        @test ctb_basis isa Basis
        @test tb_basis.name    == :∂
        @test tb_basis.type    == :coordinate
        @test tb_basis.vbundle == :tangentBA_M
        @test ctb_basis.name    == :dx
        @test ctb_basis.type    == :coordinate
        @test ctb_basis.vbundle == :cotangentBA_M

        # Variables ∂, dx, e, θ all bound in scope
        @test ∂  isa Basis
        @test dx isa Basis
        @test e  isa Basis
        @test θ  isa Basis
        @test ∂  == tb_basis
        @test dx == ctb_basis
        @test e.type  == :moving
        @test θ.type  == :moving

        # basis_for_vbundle accessor (default type=:coordinate)
        @test basis_for_vbundle(:tangentBA_M)                    == tb_basis
        @test basis_for_vbundle(:cotangentBA_M)                  == ctb_basis
        @test basis_for_vbundle(:tangentBA_M;   type=:moving)    == _BASES[(:tangentBA_M,   :moving)]
        @test basis_for_vbundle(:cotangentBA_M; type=:moving)    == _BASES[(:cotangentBA_M, :moving)]
        @test_throws ErrorException basis_for_vbundle(:no_such_bundle)

        # Equality and hashing
        @test Basis(:∂, :tangentBA_M, :coordinate) == Basis(:∂, :tangentBA_M, :coordinate)
        @test Basis(:∂, :tangentBA_M, :coordinate) != Basis(:dx, :tangentBA_M, :coordinate)
        @test Basis(:∂, :tangentBA_M, :coordinate) != Basis(:∂, :tangentBA_M, :moving)
        @test hash(Basis(:dx, :cotangentBA_M, :coordinate)) ==
              hash(Basis(:dx, :cotangentBA_M, :coordinate))

        # show does not throw
        @test (repr(tb_basis); true)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_manifold — custom frame names" begin
        _clear_all_registries!()
        @def_manifold CB_M 4 [cb_a, cb_b, cb_c, cb_d] [CB_A1, CB_A2, CB_A3, CB_A4] natural_frame=:e natural_coframe=:θ moving_frame=:ehat moving_coframe=:θhat

        @test _BASES[(:tangentCB_M,   :coordinate)].name == :e
        @test _BASES[(:cotangentCB_M, :coordinate)].name == :θ
        @test _BASES[(:tangentCB_M,   :moving)].name     == :ehat
        @test _BASES[(:cotangentCB_M, :moving)].name     == :θhat

        @test e     isa Basis
        @test θ     isa Basis
        @test ehat  isa Basis
        @test θhat  isa Basis
        @test e.type    == :coordinate
        @test θ.type    == :coordinate
        @test ehat.type == :moving
        @test θhat.type == :moving

        # FrameBundle variables bound in scope
        @test frameCB_M   isa FrameBundle
        @test coframeCB_M isa FrameBundle

        # All four frame names must be distinct → error if any duplicated
        @test_throws ErrorException macroexpand(
            @__MODULE__,
            :(@def_manifold SAME_M 4 [s_a, s_b, s_c, s_d] [S_A1, S_A2, S_A3, S_A4] natural_frame=:q natural_coframe=:q),
        )
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_frame_bundle standalone" begin
        _clear_all_registries!()
        @def_manifold FB_M 4 [fb_a, fb_b, fb_c, fb_d] [FBM_B1, FBM_B2, FBM_B3, FBM_B4]
        # @def_manifold already registered ∂/dx (coord) and e/θ (moving)
        # for tangentFB_M/cotangentFB_M.  Test standalone for a custom bundle.
        @def_vbundle FB_E FB_M 3 [FB_A1, FB_A2, FB_A3]

        # 4-arg form: frame_name vbundle_name basis_name cobasis_name
        @def_frame_bundle frameFB_E FB_E fb_e fb_edual

        @test haskey(_BASES, (:FB_E,    :moving))
        @test haskey(_BASES, (:dualFB_E, :moving))
        @test _BASES[(:FB_E,    :moving)].name == :fb_e
        @test _BASES[(:dualFB_E,:moving)].name == :fb_edual

        @test fb_e     isa Basis
        @test fb_edual isa Basis
        @test fb_e.type    == :moving
        @test fb_edual.type == :moving

        # FrameBundle variables bound in scope
        @test frameFB_E   isa FrameBundle
        @test coframeFB_E isa FrameBundle
        @test haskey(_FRAME_BUNDLES, :frameFB_E)
        @test haskey(_FRAME_BUNDLES, :coframeFB_E)

        @test frameFB_E.vbundle == :FB_E
        @test frameFB_E.dual    == :coframeFB_E
        @test frameFB_E.basis   == fb_e
        @test coframeFB_E.vbundle == :dualFB_E
        @test coframeFB_E.dual    == :frameFB_E

        # fb_edual lives in dualFB_E; index must be from FB_E (its dual)
        # FB_A1 is CoordinateIndex(:FB_A1, :FB_E) → fb_edual[FB_A1] is valid
        be = fb_edual[FB_A1]
        @test be isa BasisElement
        @test be.basis == _BASES[(:dualFB_E, :moving)]
        @test be.index.vbundle == :FB_E

        # fb_e lives in FB_E; index must be from dualFB_E → use -FB_A1
        be2 = fb_e[-FB_A1]
        @test be2 isa BasisElement
        @test be2.basis == _BASES[(:FB_E, :moving)]
        @test be2.index.vbundle == :dualFB_E

        # FrameBundle show does not throw
        @test (repr(frameFB_E); true)
        @test (repr("text/html", frameFB_E); true)

        # Argument validation: basis_name and cobasis_name must differ
        @test_throws ErrorException macroexpand(
            @__MODULE__,
            :(@def_frame_bundle frameFB_E2 FB_E same_name same_name),
        )
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "BasisElement — getindex validation" begin
        _clear_all_registries!()
        @def_manifold BE_M 4 [be_a, be_b, be_c, be_d] [BEM_B1, BEM_B2, BEM_B3, BEM_B4]
        # dx lives in cotangentBE_M; elements must be labeled by tangentBE_M (dual) indices
        # be_a is CoordinateIndex(:be_a, :tangentBE_M) → dx[be_a] is valid
        be1 = dx[be_a]
        @test be1 isa BasisElement
        @test be1.basis  == Basis(:dx, :cotangentBE_M, :coordinate)
        @test be1.index  == CoordinateIndex(:be_a, :tangentBE_M)
        @test !be1.index.is_down  # upper index (from tangentBE_M)

        # ∂ lives in tangentBE_M; elements must be labeled by cotangentBE_M (dual) indices
        # -be_a is CoordinateIndex(:be_a, :cotangentBE_M) → ∂[-be_a] is valid
        be2 = ∂[-be_a]
        @test be2 isa BasisElement
        @test be2.basis  == Basis(:∂, :tangentBE_M, :coordinate)
        @test be2.index  == CoordinateIndex(:be_a, :cotangentBE_M)
        @test be2.index.is_down   # lower index (from cotangentBE_M)

        # Wrong vbundle → error:
        # dx lives in cotangentBE_M, dual is tangentBE_M → requires tangentBE_M index
        # -be_a is cotangentBE_M, NOT tangentBE_M → invalid
        @test_throws ErrorException dx[-be_a]
        # ∂ lives in tangentBE_M, dual is cotangentBE_M → requires cotangentBE_M index
        # be_a is tangentBE_M, NOT cotangentBE_M → invalid
        @test_throws ErrorException ∂[be_a]

        # Wrong index kind (coordinate vs basis) for frame type
        @test_throws ErrorException e[-be_a]
        @test_throws ErrorException e[be_a]
        @test_throws ErrorException θ[be_a]
        @test_throws ErrorException dx[BEM_B1]
        @test_throws ErrorException ∂[-BEM_B1]

        # Correct moving-frame elements
        be3 = e[-BEM_B1]
        @test be3 isa BasisElement
        @test be3.index isa BasisIndex
        be4 = θ[BEM_B1]
        @test be4 isa BasisElement
        @test be4.index isa BasisIndex

        # Equality and hashing
        @test be1 == dx[be_a]
        @test hash(be1) == hash(dx[be_a])
        @test be1 != be2

        # Basis type field in assertions
        @test be1.basis == Basis(:dx, :cotangentBE_M, :coordinate)
        @test be2.basis == Basis(:∂,  :tangentBE_M,   :coordinate)

        # show does not throw
        @test (repr(be1); true)
        @test (repr("text/latex", be1); true)
        @test (repr("text/html",  be1); true)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "basis_expansion — covariant tensor" begin
        _clear_all_registries!()
        @def_manifold COV_M 4 [cov_a, cov_b, cov_c, cov_d] [COVM_B1, COVM_B2, COVM_B3, COVM_B4]
        @def_metric cov_g COV_M
        @def_tensor COV_T [cotangentCOV_M, cotangentCOV_M]

        # Via TensorExpression
        te  = COV_T[-cov_a, -cov_b]
        bx  = basis_expansion(te)

        @test bx isa BasisExpansion
        @test bx.component === te
        @test length(bx.basis_elements) == 2

        be1, be2 = bx.basis_elements
        # Covariant slot → cotangentCOV_M → basis is dx, index from tangentCOV_M
        @test be1.basis.name    == :dx
        @test be1.index.symbol  == :cov_a
        @test !be1.index.is_down   # upper index (from tangentCOV_M)

        @test be2.basis.name    == :dx
        @test be2.index.symbol  == :cov_b
        @test !be2.index.is_down

        # Via Tensor directly
        bx2 = basis_expansion(COV_T)
        @test bx2 isa BasisExpansion
        @test length(bx2.basis_elements) == 2

        # show does not throw
        @test (repr(bx); true)
        @test (repr("text/latex", bx); true)
        @test (repr("text/html",  bx); true)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "basis_expansion — contravariant tensor" begin
        _clear_all_registries!()
        @def_manifold CTR_M 4 [ctr_a, ctr_b, ctr_c, ctr_d] [CTRM_B1, CTRM_B2, CTRM_B3, CTRM_B4]
        @def_tensor CTR_T [tangentCTR_M, tangentCTR_M]

        te = CTR_T[ctr_a, ctr_b]
        bx = basis_expansion(te)

        @test length(bx.basis_elements) == 2
        be1, be2 = bx.basis_elements

        # Contravariant slot → tangentCTR_M → basis is ∂, index from cotangentCTR_M
        @test be1.basis.name   == :∂
        @test be1.index.is_down    # lower index (from cotangentCTR_M)
        @test be2.basis.name   == :∂
        @test be2.index.is_down
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "basis_expansion — mixed tensor" begin
        _clear_all_registries!()
        @def_manifold MX_M 4 [mx_a, mx_b, mx_c, mx_d] [MXM_B1, MXM_B2, MXM_B3, MXM_B4]
        @def_tensor MX_T [tangentMX_M, cotangentMX_M]

        te = MX_T[mx_a, -mx_b]
        bx = basis_expansion(te)

        @test length(bx.basis_elements) == 2
        be1, be2 = bx.basis_elements

        # First slot: contravariant → tangentMX_M → ∂ with lower index
        @test be1.basis.name  == :∂
        @test be1.index.is_down

        # Second slot: covariant → cotangentMX_M → dx with upper index
        @test be2.basis.name  == :dx
        @test !be2.index.is_down
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "VBundle.bases virtual property" begin
        _clear_all_registries!()
        @def_manifold VP_M 4 [vp_a, vp_b, vp_c, vp_d] [VPM_B1, VPM_B2, VPM_B3, VPM_B4]

        @test tangentVP_M.bases   isa Vector{Basis}
        @test cotangentVP_M.bases isa Vector{Basis}
        @test length(tangentVP_M.bases)   == 2
        @test length(cotangentVP_M.bases) == 2

        @test tangentVP_M.bases[1].name    == :∂
        @test tangentVP_M.bases[1].type    == :coordinate
        @test tangentVP_M.bases[2].name    == :e
        @test tangentVP_M.bases[2].type    == :moving

        @test length(tangentVP_M.basis_indices) == 4
        @test tangentVP_M.basis_indices[1].symbol == :VPM_B1
        @test :basis_indices in propertynames(tangentVP_M)
        @test cotangentVP_M.bases[1].name  == :dx
        @test cotangentVP_M.bases[1].type  == :coordinate
        @test cotangentVP_M.bases[2].name  == :θ
        @test cotangentVP_M.bases[2].type  == :moving

        @test tangentVP_M.bases[1].vbundle   == :tangentVP_M
        @test cotangentVP_M.bases[1].vbundle == :cotangentVP_M

        # :bases in propertynames
        @test :bases in propertynames(tangentVP_M)
        @test :bases in propertynames(cotangentVP_M)
        @test :basis ∉ propertynames(tangentVP_M)

        # Bundle with no frames registered → empty vector
        @def_vbundle VP_E VP_M 2 [VP_v1, VP_v2]
        @test VP_E.bases == Basis[]
        @test dualVP_E.bases == Basis[]

        # Standalone frame bundle adds moving basis to vbundle.bases
        @def_frame_bundle frameVP_E VP_E vp_e vp_theta
        @test length(VP_E.bases) == 1
        @test VP_E.bases[1].name == :vp_e
        @test VP_E.bases[1].type == :moving
        @test length(dualVP_E.bases) == 1
        @test dualVP_E.bases[1].name == :vp_theta
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@undef_manifold cleans _BASES and _FRAME_BUNDLES" begin
        _clear_all_registries!()
        @def_manifold UM_BX_M 4 [umbx_a, umbx_b, umbx_c, umbx_d] [UMBXM_B1, UMBXM_B2, UMBXM_B3, UMBXM_B4]

        @test haskey(_BASES, (:tangentUM_BX_M,   :coordinate))
        @test haskey(_BASES, (:cotangentUM_BX_M, :coordinate))
        @test haskey(_BASES, (:tangentUM_BX_M,   :moving))
        @test haskey(_BASES, (:cotangentUM_BX_M, :moving))
        @test haskey(_FRAME_BUNDLES, :frameUM_BX_M)
        @test haskey(_FRAME_BUNDLES, :coframeUM_BX_M)

        @undef_manifold UM_BX_M
        @test !haskey(_BASES, (:tangentUM_BX_M,   :coordinate))
        @test !haskey(_BASES, (:cotangentUM_BX_M, :coordinate))
        @test !haskey(_BASES, (:tangentUM_BX_M,   :moving))
        @test !haskey(_BASES, (:cotangentUM_BX_M, :moving))
        @test !haskey(_FRAME_BUNDLES, :frameUM_BX_M)
        @test !haskey(_FRAME_BUNDLES, :coframeUM_BX_M)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "@def_vbundle — no basis= / cobasis= kwargs (removed)" begin
        _clear_all_registries!()
        @def_manifold VBF_M 4 [vbf_a, vbf_b, vbf_c, vbf_d] [VBFM_B1, VBFM_B2, VBFM_B3, VBFM_B4]
        @def_vbundle VBF_E VBF_M 3 [VBF_A1, VBF_A2, VBF_A3]

        # No coordinate or moving frames registered for standalone vbundle
        @test !haskey(_BASES, (:VBF_E,    :coordinate))
        @test !haskey(_BASES, (:dualVBF_E,:coordinate))
        @test VBF_E.bases    == Basis[]
        @test dualVBF_E.bases == Basis[]

        # basis= kwarg is no longer supported → error at macroexpand time
        @test_throws ErrorException macroexpand(
            @__MODULE__,
            :(@def_vbundle VBF_E2 VBF_M 3 [VBF_B1, VBF_B2, VBF_B3] basis=:myframe),
        )
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "FrameBundle struct and registry" begin
        _clear_all_registries!()
        @def_manifold FR_M 4 [fr_a, fr_b, fr_c, fr_d] [FRM_B1, FRM_B2, FRM_B3, FRM_B4]

        # @def_manifold binds frameM, coframeM, e, θ automatically
        @test frameFR_M   isa FrameBundle
        @test coframeFR_M isa FrameBundle
        @test haskey(_FRAME_BUNDLES, :frameFR_M)
        @test haskey(_FRAME_BUNDLES, :coframeFR_M)

        @test frameFR_M.name    == :frameFR_M
        @test frameFR_M.vbundle == :tangentFR_M
        @test frameFR_M.dual    == :coframeFR_M
        @test frameFR_M.basis   isa Basis
        @test frameFR_M.basis.name == :e
        @test frameFR_M.basis.type == :moving

        @test coframeFR_M.name    == :coframeFR_M
        @test coframeFR_M.vbundle == :cotangentFR_M
        @test coframeFR_M.dual    == :frameFR_M
        @test coframeFR_M.basis.name == :θ
        @test coframeFR_M.basis.type == :moving

        # e and θ bound as Basis (moving)
        @test e isa Basis
        @test θ isa Basis
        @test e.type == :moving
        @test θ.type == :moving

        # Moving frame accessible via _BASES with :moving key
        @test _BASES[(:tangentFR_M,   :moving)] == e
        @test _BASES[(:cotangentFR_M, :moving)] == θ

        # Coordinate frame still accessible
        @test ∂  isa Basis
        @test dx isa Basis
        @test ∂.type  == :coordinate
        @test dx.type == :coordinate

        # Equality and hashing for FrameBundle
        fb_copy = FrameBundle(:frameFR_M, :tangentFR_M, :coframeFR_M, e)
        @test frameFR_M == fb_copy
        @test hash(frameFR_M) == hash(fb_copy)

        # show does not throw
        @test (repr(frameFR_M); true)
        @test (repr("text/html", frameFR_M); true)

        # propertynames
        @test :basis in propertynames(frameFR_M)
    end


    # ─────────────────────────────────────────────────────────────────
    @testset "basis_expansion — frame=:moving" begin
        _clear_all_registries!()
        @def_manifold MV_M 4 [mv_a, mv_b, mv_c, mv_d] [MVM_B1, MVM_B2, MVM_B3, MVM_B4]
        @def_metric mv_g MV_M
        @def_tensor MV_T [cotangentMV_M, cotangentMV_M]

        te = MV_T[-mv_a, -mv_b]

        # Default (coordinate)
        bx_coord = basis_expansion(te)
        @test bx_coord.basis_elements[1].basis.name == :dx
        @test bx_coord.basis_elements[1].basis.type == :coordinate

        # Moving frame
        bx_mov = basis_expansion(te; frame=:moving)
        @test bx_mov isa BasisExpansion
        @test bx_mov.component === te
        @test length(bx_mov.basis_elements) == 2

        be1, be2 = bx_mov.basis_elements
        @test be1.basis.name   == :θ
        @test be1.basis.type   == :moving
        @test be2.basis.name   == :θ
        @test !be1.index.is_down  # upper index (from tangentMV_M)

        # Via Tensor overload
        bx_t = basis_expansion(MV_T; frame=:moving)
        @test bx_t isa BasisExpansion
        @test bx_t.basis_elements[1].basis.type == :moving

        # show does not throw for moving expansion
        @test (repr(bx_mov); true)
        @test (repr("text/latex", bx_mov); true)
        @test (repr("text/html",  bx_mov); true)

        # Unknown frame type → error
        @test_throws ErrorException basis_expansion(te; frame=:unknown_frame)
    end

end # @testset "AbstractTensors.jl"
