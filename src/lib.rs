#[derive(Debug)]
pub struct InitSeqEstimate {
    var_pos: f64,
    var_dec: f64,
    var_con: f64,
    gamma_pos: Vec<f64>,
    gamma_dec: Vec<f64>,
    gamma_con: Vec<f64>,
}

#[inline]
fn offset_dot_self(x: &[f64], lb: usize, ub: usize) -> f64 {
    // Option 1
    x[..ub]
        .iter()
        .zip(x[lb..].iter())
        .map(|(a, b)| *a * *b)
        .sum::<f64>()
    // Option 2
    // x[..ub]
    //     .iter()
    //     .zip(x[lb..].iter())
    //     .fold(0.0, |acc, (a, b)| a.mul_add(*b, acc))
}

pub fn init_seq(x: &[f64]) -> InitSeqEstimate {
    let n = x.len();
    let inv_n = 1.0 / n as f64;
    let mu = x.iter().sum::<f64>() * inv_n;
    let z: Vec<f64> = x.iter().map(move |x_i| *x_i - mu).collect();

    let half = n / 2;
    let mut gamma_hat: Vec<f64> = Vec::with_capacity(half);
    let mut gamma_0: f64 = 0.0;
    for k in 0..half {
        let lb = 2 * k;
        let ub = n - lb;
        let gamma_2k = offset_dot_self(&z, lb, ub) * inv_n;
        if k == 0 {
            gamma_0 = gamma_2k;
        }

        let lb = lb + 1;
        let ub = ub - 1;
        let gamma_2k1 = offset_dot_self(&z, lb, ub) * inv_n;

        let gamma_hat_k = gamma_2k + gamma_2k1;
        if gamma_hat_k > 0.0 {
            gamma_hat.push(gamma_hat_k);
        } else {
            gamma_hat.push(0.0);
            break;
        }
    }

    let m = gamma_hat.len();
    let mut gamma_pos: Vec<f64> = Vec::with_capacity(m);
    let mut gamma_dec: Vec<f64> = Vec::with_capacity(m);
    let mut min = f64::MAX;
    for gamma_hat_k in gamma_hat.iter_mut() {
        gamma_pos.push(*gamma_hat_k);
        if *gamma_hat_k > min {
            *gamma_hat_k = min;
        } else {
            min = gamma_hat_k.clone();
        }
        gamma_dec.push(min);
    }

    // Greatest convex minorant via isotonic regression on derivative
    for k in (1..m).into_iter().rev() {
        gamma_hat[k] -= gamma_hat[k - 1];
    }

    // Pool Adjacent Violator Algorithm
    let mut p: Vec<f64> = vec![0.0; m];
    let mut nu: Vec<usize> = vec![0; m];
    let mut n: usize = 0;
    for k in 1..m {
        p[n] = gamma_hat[k];
        nu[n] = 1;
        n += 1;
        while n > 1 && (p[n - 1] / nu[n - 1] as f64) < (p[n - 2] / nu[n - 2] as f64) {
            p[n - 2] += p[n - 1];
            nu[n - 2] += nu[n - 1];
            n -= 1;
        }
    }
    let mut k: usize = 1;
    for (p_j, nu_j) in p[0..n].iter().zip(nu[0..n].iter()) {
        let mu = *p_j / *nu_j as f64;
        for _ in 0..*nu_j {
            gamma_hat[k] = gamma_hat[k - 1] + mu;
            k += 1;
        }
    }
    let gamma_con: Vec<f64> = gamma_hat.into_iter().collect();

    let var_pos = 2.0 * gamma_pos.iter().sum::<f64>() - gamma_0;
    let var_dec = 2.0 * gamma_dec.iter().sum::<f64>() - gamma_0;
    let var_con = 2.0 * gamma_con.iter().sum::<f64>() - gamma_0;

    InitSeqEstimate {
        var_pos,
        var_dec,
        var_con,
        gamma_pos,
        gamma_dec,
        gamma_con,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
