#[derive(Debug)]
pub struct InitSeq {
    var_pos: f64,
    var_dec: f64,
    var_con: f64,
    gamma_pos: Vec<f64>,
    gamma_dec: Vec<f64>,
    gamma_con: Vec<f64>,
}
pub enum Estimator {
    Positive,
    Monotone,
    Convex,
}
use self::Estimator::*;
impl InitSeq {
    pub fn var(&self, rhs: Estimator) -> f64 {
        match rhs {
            Positive => self.var_pos,
            Monotone => self.var_dec,
            Convex => self.var_con,
        }
    }
    pub fn gamma<'a>(&'a self, rhs: Estimator) -> &'a Vec<f64> {
        match rhs {
            Positive => &self.gamma_pos,
            Monotone => &self.gamma_dec,
            Convex => &self.gamma_con,
        }
    }
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

pub fn init_seq(x: &[f64]) -> InitSeq {
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

    // // Greatest convex minorant via isotonic regression on derivative
    // for k in (1..m).into_iter().rev() {
    //     gamma_hat[k] -= gamma_hat[k - 1];
    // }

    // // Pool Adjacent Violator Algorithm
    // let mut p: Vec<f64> = Vec::with_capacity(m);
    // p.resize(m, 0.0);
    // let mut nu: Vec<usize> = Vec::with_capacity(m);
    // nu.resize(m, 0);
    // let mut n: usize = 0;
    // for k in 1..m {
    //     p[n] = gamma_hat[k];
    //     nu[n] = 1;
    //     n += 1;
    //     while n > 1 && (p[n - 1] / nu[n - 1] as f64) < (p[n - 2] / nu[n - 2] as f64) {
    //         p[n - 2] += p[n - 1];
    //         nu[n - 2] += nu[n - 1];
    //         n -= 1;
    //     }
    // }
    // let mut k: usize = 1;
    // for (p_j, nu_j) in p[0..n].iter().zip(nu[0..n].iter()) {
    //     let mu = *p_j / *nu_j as f64;
    //     for _ in 0..*nu_j {
    //         gamma_hat[k] = gamma_hat[k - 1] + mu;
    //         k += 1;
    //     }
    // }
    // p.truncate(n - 1);
    // nu.truncate(n - 1);
    // for (p_j, nu_j) in p.into_iter().zip(nu.into_iter()) {
    //     let mu = p_j / nu_j as f64;
    //     for _ in 0..nu_j {
    //         gamma_hat[k] = gamma_hat[k - 1] + mu;
    //         k += 1;
    //     }
    // }
    // let gamma_con: Vec<f64> = gamma_hat.into_iter().collect();

    // Greatest convex minorant via isotonic regression on derivative
    for k in (1..m).into_iter().rev() {
        gamma_hat[k] -= gamma_hat[k - 1];
    }
    // let mut k = m - 1;
    // while k > 0 {
    //     gamma_hat[k] -= gamma_hat[k - 1];
    //     k -= 1;
    // }
    let n = gamma_hat.len() - 1;
    let mut nu: Vec<f64> = Vec::with_capacity(n);
    nu.push(gamma_hat[1]);
    let mut w: Vec<usize> = Vec::with_capacity(n);
    w.push(1);
    let mut j: usize = 0;
    let mut i: usize = 1;
    while i < n {
        j += 1;
        nu.push(gamma_hat[i + 1]);
        w.push(1);
        i += 1;
        while j > 0 && nu[j - 1] > nu[j] {
            let w_prime = w[j - 1] + w[j];
            let nu_prime = (w[j - 1] as f64 * nu[j - 1] + w[j] as f64 * nu[j]) / w_prime as f64;
            nu[j - 1] = nu_prime;
            w[j - 1] = w_prime;
            nu.swap_remove(j);
            w.swap_remove(j);
            j -= 1;
        }
    }
    let m = j + 1;
    j = 0;
    gamma_hat.shrink_to_fit();
    let mut gamma_con = gamma_hat;
    let mut pos: usize = 1;
    while j < m {
        let mu = nu[j];
        for _ in 0..w[j] {
            gamma_con[pos] = gamma_con[pos - 1] + mu;
            pos += 1;
        }
        j += 1;
    }

    let var_pos = 2.0 * gamma_pos.iter().sum::<f64>() - gamma_0;
    let var_dec = 2.0 * gamma_dec.iter().sum::<f64>() - gamma_0;
    let var_con = 2.0 * gamma_con.iter().sum::<f64>() - gamma_0;

    InitSeq {
        var_pos,
        var_dec,
        var_con,
        gamma_pos,
        gamma_dec,
        gamma_con,
    }
}
// fn diff(x: &[f64]) -> Vec<f64> {
//     let n = x.len();
//     let mut dx: Vec<f64> = Vec::with_capacity(n - 1);

//     for i in 0..n - 1 {
//         // We know this is in bounds given that the length is n, and
//         // (n - 2 + 1) = n - 1 is the last offset accessed.
//         let delta = unsafe { *x.get_unchecked(i + 1) - *x.get_unchecked(i) };
//         dx.push(delta);
//     }
//     dx
// }

#[cfg(test)]
mod tests {
    use super::*;
}
