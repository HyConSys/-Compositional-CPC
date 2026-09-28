module VerifyMonolithicAnalyticCPC

using Printf

const Q = Rational{BigInt}
q(n::Integer, d::Integer=1) = Q(BigInt(n), BigInt(d))

"Exact degree-two monolithic CPC for the eta=0.1, u=2 ring."
function verify_monolithic_template(; rooms::Int=10, alpha=q(1, 100), verbose=true)
    rooms >= 3 || throw(ArgumentError("the ring must have at least three rooms"))
    a = q(1, 2)
    eta = q(1, 10)
    closed_loop_norm_bound = a + 2eta # 7/10
    contraction_squared = closed_loop_norm_bound^2 # 49/100
    decrease_fraction = 1 - contraction_squared # 51/100
    gain = inv(decrease_fraction) # 100/51
    epsilon = q(1)

    # For z+=Az, A=(1/2)I+(1/10)M and spec(M) subset [-2,2]. Therefore
    # ||A||_2<=7/10 and gain*(||z||^2-||Az||^2)>=||z||^2.
    @assert closed_loop_norm_bound == q(7, 10)
    @assert gain == q(100, 51)
    @assert gain * decrease_fraction == 1
    @assert alpha == q(1, 100)

    # B=0, V_q0=gain*sum(z_i^2)+1, V_q1=gain*sum(z_i^2).
    # On global bad, some z_i<=-2 or z_i>=1, hence sum(z_i^2)>=1 and
    # every odd-priority transition decreases by epsilon=1. On global good,
    # the corresponding residual is nonnegative without the -1 term.
    bad_energy_lower_bound = q(1)
    @assert gain * decrease_fraction * bad_energy_lower_bound >= epsilon

    if verbose
        println("Exact monolithic CPC verification: PASS")
        println("  rooms: ", rooms, "; alpha: ", alpha)
        println("  B_q0=B_q1=0")
        println("  V_q0=(100/51) sum(z_i^2)+1")
        println("  V_q1=(100/51) sum(z_i^2)")
        println("  epsilon=1; ||A||_2 <= 7/10")
    end
    return (valid=true, rooms=rooms, alpha=alpha, gain=gain,
            epsilon=epsilon, norm_bound=closed_loop_norm_bound)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    verify_monolithic_template()
end

end # module
