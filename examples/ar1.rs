use initseq::{init_seq, Estimator};
use std::time::SystemTime;

// N.B. This is a notoriously poor method for seeding a PRNG.
// For a single PRNG, used once, it is probably acceptable, but in general,
// this will produce seeds which are very similar.
fn bad_seed() -> u64 {
    let now = SystemTime::now();
    let elapsed = now.duration_since(SystemTime::UNIX_EPOCH).unwrap();
    (elapsed.as_nanos() as u64).wrapping_mul(0xbad_c0ffee_bad_cafe)
}

#[derive(Debug)]
pub struct LCG {
    x: u64,
    a: u64,
    c: u64,
}
impl LCG {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            x: seed,
            a: 0xc2ec33ef97a5,
            c: 13,
        }
    }
    pub fn from_fixed_seed() -> Self {
        Self::from_seed(0xbad_c0ffee_bad_cafe)
    }
    pub fn from_default_seed() -> Self {
        Self::from_seed(bad_seed())
    }
    pub fn gen(&mut self) -> u64 {
        let x_n = self.x.wrapping_mul(self.a).wrapping_add(self.c);
        self.x = x_n;
        x_n
    }
    pub fn gen_f64(&mut self) -> f64 {
        let x_n = self.gen();
        (x_n >> 11) as f64 * 1.1102230246251565e-16
    }
    pub fn marsaglia_polar(&mut self) -> (f64, f64) {
        let mut s = 0.0;
        let mut x = 0.0;
        let mut y = 0.0;
        while s >= 1.0 || s == 0.0 {
            x = self.gen_f64().mul_add(2.0, -1.0);
            y = self.gen_f64().mul_add(2.0, -1.0);
            s = x * x + y * y;
        }
        let norm = (-2.0 * s.ln() / s).sqrt();
        (x * norm, y * norm)
    }
}

pub struct Ar1 {
    rho: f64,
    tau: f64,
    x_n: f64,
    rng: LCG,
    n_trans: usize,
}
impl Iterator for Ar1 {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        let (z, _) = self.rng.marsaglia_polar();
        let x_np1 = self.rho * self.x_n + z * self.tau;
        self.n_trans += 1;
        self.x_n = x_np1;
        Some(x_np1)
    }
}
impl Ar1 {
    // A bit heavy-handed, but this is just an example.
    pub fn from(rho: f64, tau: f64, x_n: f64) -> Self {
        Self {
            rho: rho % 1.0,
            tau: tau.abs(),
            x_n,
            rng: LCG::from_fixed_seed(),
            n_trans: 0,
        }
    }
}

fn main() {
    let rho: f64 = 0.99;
    let tau: f64 = 1.0;
    let x_0: f64 = 0.0;
    let ar1 = Ar1::from(rho, tau, x_0);

    let n: usize = 10_000;

    let x: Vec<f64> = ar1.take(n).collect();

    let mu = x.iter().sum::<f64>() / x.len() as f64;
    let var = x
        .iter()
        .map(|x_i| {
            let delta = *x_i - mu;
            delta * delta
        })
        .sum::<f64>()
        / x.len() as f64;
    let invariant_var = (tau * tau) / (1.0 - rho * rho);
    let asymptotic_var = invariant_var * (1.0 + rho) / (1.0 - rho);

    let initial = init_seq(&x);
    let words = vec!["positive", "monotone", "convex"];
    let estimators = vec![Estimator::Positive, Estimator::Monotone, Estimator::Convex];

    println!("{:-^80}", "Asymptotic variance estimates");
    for (word, estimator) in words.iter().zip(estimators.iter()) {
        println!("Initial {} sequence: {}", word, initial.var(*estimator));
    }
    println!("Variance as if i.i.d. (far from true here): {}", var);

    println!("Variance of invariant distribution: {}", invariant_var);
    println!("Asymptotic variance: {}", asymptotic_var);
    for (word, estimator) in words.iter().zip(estimators.iter()) {
        println!(
            "Effective sample size, {}: {}",
            word,
            (var / initial.var(*estimator)) * n as f64
        );
    }

    println!("{:-^80}", "Autocovariance sequences");
    for (word, estimator) in words.iter().zip(estimators.iter()) {
        println!("Initial {} sequence: {:?}", word, initial.gamma(*estimator));
    }
}
