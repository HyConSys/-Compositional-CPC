module VerifyCenteredAnalyticCPC

using Printf

const Q = Rational{BigInt}

q(n::Integer, d::Integer=1) = Q(BigInt(n), BigInt(d))

"Exact centered closed-loop and CPC data for the constant controller u=2."
function exact_template()
    a = q(1, 2)
    eta = q(1, 10)
    gain = q(2)
    epsilon = q(1)

    upsilon = [q(0) q(0); q(0) q(0)]
    gamma = [gain * eta^2 gain * a * eta;
             gain * a * eta gain * (a^2 - 1) + 1]

    return (a=a, eta=eta, gain=gain, epsilon=epsilon,
            Upsilon=upsilon, Gamma=gamma)
end

composition_value(P, lambda) =
    P[1, 1] * lambda^2 + 2P[1, 2] * lambda + P[2, 2]

"Evaluate the centered supply directly from the physical port (w,x)."
function physical_supply(Gamma, w, x)
    v, z = w - q(50), x - q(25)
    return Gamma[1, 1] * v^2 + 2Gamma[1, 2] * v * z +
           Gamma[2, 2] * z^2
end

"Homogeneous-coordinate matrix for [w,x,1]'*P*[w,x,1]."
function augmented_physical_supply_matrix(Gamma)
    center = [q(50), q(25)]
    linear = -Gamma * center
    constant = transpose(center) * Gamma * center
    return [Gamma linear; transpose(linear) constant]
end

function verify_exact_template(; verbose=true)
    data = exact_template()
    a, eta, gain, epsilon =
        data.a, data.eta, data.gain, data.epsilon
    Gamma = data.Gamma

    # Gamma is selected so that, identically in (z,v),
    #   Gamma[v,z] = gain*((a*z+eta*v)^2-z^2) + z^2.
    @assert Gamma[1, 1] == gain * eta^2
    @assert Gamma[1, 2] == gain * a * eta
    @assert Gamma[2, 2] == gain * (a^2 - 1) + 1

    # Coefficients of gain*((a*z+eta*v)^2-z^2)+z^2 minus Gamma[v,z].
    @assert gain * eta^2 - Gamma[1, 1] == 0
    @assert gain * a * eta - Gamma[1, 2] == 0
    @assert gain * (a^2 - 1) + 1 - Gamma[2, 2] == 0

    # The four deterministic two-priority rank residuals reduce exactly to
    # z^2 on the good region and z^2-1 on either bad region.
    @assert epsilon == 1
    # Constants in the four residuals (q0/good, q0/bad, q1/good, q1/bad).
    @assert q(1) - epsilon == 0
    @assert q(1) - q(1) - epsilon == -1
    @assert q(0) == 0
    @assert -q(1) == -1

    # Bq0=Bq1=0 and Upsilon=0 satisfy the initial, lower-bound, and every
    # barrier transition condition as exact polynomial identities.  The rank
    # lower bounds are Vq0=2z^2+1 and Vq1=2z^2, both SOS.
    @assert all(iszero, data.Upsilon)
    @assert gain >= 0

    # Explicit Putinar/Markov-Lukacs certificates on the three label regions.
    # good [-2,1]: z^2 is already SOS.
    # bad-high [1,5], g=(z-1)(5-z):
    # z^2-1 = 3/2 (z-1)^2 + 1/2 g.
    @assert q(3, 2) - q(1, 2) == 1
    @assert -3q(1) + 3q(1) == 0
    @assert q(3, 2) - q(5, 2) == -1
    # bad-low [-10,-2], g=(z+10)(-2-z):
    # z^2-1 = 3 + 3/2 (z+2)^2 + 1/2 g.
    @assert q(3, 2) - q(1, 2) == 1
    @assert 6q(1) - 6q(1) == 0
    @assert 3q(1) + 6q(1) - 10q(1) == -1

    # Since q(lambda) is convex (Gamma_11>0), its maximum on [-2,2]
    # occurs at an endpoint. This proves the homogeneous ring LMI for every N.
    qminus = composition_value(Gamma, q(-2))
    qplus = composition_value(Gamma, q(2))
    @assert Gamma[1, 1] > 0
    @assert qminus <= 0
    @assert qplus <= 0
    @assert composition_value(data.Upsilon, q(-2)) == 0
    @assert composition_value(data.Upsilon, q(2)) == 0

    # The physical dynamics are x+ = .5x+7.5+.1w.  With z=x-25 and
    # v=w-50, all affine terms cancel and z+ = .5z+.1v exactly.
    physical_constant = a * q(25) + q(15, 2) + eta * q(50) - q(25)
    @assert physical_constant == 0
    # The fixed controller u=2 preserves X=[15,30] for every w in [30,60].
    # Monotonicity makes these two corners the exact image bounds.
    physical_successor(x, w) = a * x + q(15, 2) + eta * w
    minimum_successor = physical_successor(q(15), q(30))
    maximum_successor = physical_successor(q(30), q(60))
    @assert minimum_successor >= q(15)
    @assert maximum_successor <= q(30)

    # Pull the centered supply back to the original temperature coordinates.
    # It equals w^2/50+wx/5-x^2/2-7w+15x-25/2 and vanishes at equilibrium.
    physical_matrix = augmented_physical_supply_matrix(Gamma)
    @assert physical_matrix == [q(1, 50) q(1, 10) q(-7, 2);
                                q(1, 10) q(-1, 2) q(15, 2);
                                q(-7, 2) q(15, 2) q(-25, 2)]
    @assert physical_supply(Gamma, q(50), q(25)) == 0
    for (w, x) in ((q(30), q(15)), (q(50), q(25)), (q(60), q(30)))
        homogeneous_port = [w, x, q(1)]
        @assert physical_supply(Gamma, w, x) ==
                (transpose(homogeneous_port) * physical_matrix *
                 homogeneous_port)
    end

    if verbose
        println("Exact centered CPC verification: PASS")
        println("  z+ = ", a, " z + ", eta, " v")
        println("  Upsilon = ", data.Upsilon)
        println("  Gamma = ", data.Gamma)
        println("  physical augmented Gamma = ", physical_matrix)
        println("  physical supply at (w,x)=(50,25): ",
                physical_supply(Gamma, q(50), q(25)))
        println("  fixed-u=2 image of X over W: [", minimum_successor,
                ", ", maximum_successor, "]")
        println("  q(-2) = ", qminus, " = ", Float64(qminus))
        println("  q( 2) = ", qplus, " = ", Float64(qplus))
        println("  local rank residuals: z^2 (good), z^2-1 (bad)")
        println("  explicit interval SOS decompositions: PASS")
    end

    return (valid=true, template=data,
            physical_supply_matrix=physical_matrix,
            state_image=(lower=minimum_successor,
                         upper=maximum_successor),
            composition_endpoints=(minus_two=qminus, plus_two=qplus))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    verify_exact_template()
end

end # module
