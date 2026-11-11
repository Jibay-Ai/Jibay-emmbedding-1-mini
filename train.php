<?php
set_time_limit(60);
ignore_user_abort(true);
ini_set('memory_limit', '256M');

$ROOT = __DIR__;
$DATA = $ROOT . '/dataset.txt';
$TOKENIZER = $ROOT . '/tokenizer.json';
$TOKENIZER_CONFIG = $ROOT . '/tokenizer_config.json';
$SPECIAL = $ROOT . '/special_tokens_map.json';
$CONFIG = $ROOT . '/config.json';
$WEIGHTS = $ROOT . '/model.weights';
$OPTIMIZER = $ROOT . '/optimizer.state';
$CHECKPOINT = $ROOT . '/checkpoint.json';
$BPE_STATE = $ROOT . '/tokenizer_build.state';
$LOCK = $ROOT . '/train.lock';

const VOCAB_TARGET = 512;
const MAX_LEN = 64;
const HIDDEN = 32;
const HEADS = 2;
const FFN = 64;
const OUT_DIM = 128;
const BATCH_PAIRS = 4;
const TIME_BUDGET = 50.0;
const MODEL_NAME = 'jibay-emmbedding';
const ARCH_NAME = 'JibayForEmmbeddingLM';
const PAD_ID = 256;
const BOS_ID = 257;
const EOS_ID = 258;
const EN_ID = 259;
const FA_ID = 260;
const FIRST_MERGE_ID = 261;

function json_out($data, $code = 200) {
    if (PHP_SAPI !== 'cli') {
        http_response_code($code);
        header('Content-Type: application/json; charset=utf-8');
    }
    echo json_encode($data, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT) . PHP_EOL;
    exit;
}

function atomic_write($path, $contents) {
    $tmp = $path . '.tmp';
    if (file_put_contents($tmp, $contents, LOCK_EX) === false) {
        throw new RuntimeException('Cannot write ' . basename($path));
    }
    if (!rename($tmp, $path)) {
        @unlink($tmp);
        throw new RuntimeException('Cannot replace ' . basename($path));
    }
}

function save_serialized($path, $value) {
    atomic_write($path, serialize($value));
}

function load_serialized($path) {
    if (!is_file($path)) return null;
    $raw = file_get_contents($path);
    if ($raw === false) return null;
    $value = @unserialize($raw, ['allowed_classes' => false]);
    return is_array($value) ? $value : null;
}

function read_pair($fp) {
    $fa = null;
    while (($line = fgets($fp)) !== false) {
        $line = trim($line, "\r\n");
        if ($line !== '') {
            $fa = trim($line);
            break;
        }
    }
    if ($fa === null || $fa === '') return null;
    $en = null;
    while (($line = fgets($fp)) !== false) {
        $line = trim($line, "\r\n");
        if ($line !== '') {
            $en = trim($line);
            break;
        }
    }
    if ($en === null || $en === '') return null;
    return [$fa, $en, ftell($fp)];
}

function byte_ids($text) {
    if ($text === '') return [];
    $u = unpack('C*', $text);
    return $u ? array_values($u) : [];
}

function read_bpe_sample($path, $limit) {
    $fp = fopen($path, 'rb');
    if (!$fp) throw new RuntimeException('Cannot open dataset.txt');
    $seqs = [];
    $pairs = 0;
    while ($pairs < $limit && ($pair = read_pair($fp)) !== null) {
        $a = byte_ids($pair[0]);
        $b = byte_ids($pair[1]);
        if (count($a) > 0) $seqs[] = $a;
        if (count($b) > 0) $seqs[] = $b;
        $pairs++;
    }
    fclose($fp);
    if ($pairs < 8) throw new RuntimeException('dataset.txt must contain at least 8 valid Persian-English pairs');
    return ['sequences' => $seqs, 'pairs' => $pairs, 'merges' => [], 'next_id' => FIRST_MERGE_ID];
}

function count_bpe_pairs($sequences) {
    $counts = [];
    foreach ($sequences as $seq) {
        $n = count($seq);
        for ($i = 0; $i + 1 < $n; $i++) {
            $key = ((int)$seq[$i]) * 1024 + (int)$seq[$i + 1];
            if (isset($counts[$key])) $counts[$key]++;
            else $counts[$key] = 1;
        }
    }
    return $counts;
}

function apply_bpe_merge($sequences, $left, $right, $newId) {
    $out = [];
    foreach ($sequences as $seq) {
        $next = [];
        $n = count($seq);
        for ($i = 0; $i < $n; $i++) {
            if ($i + 1 < $n && $seq[$i] === $left && $seq[$i + 1] === $right) {
                $next[] = $newId;
                $i++;
            } else {
                $next[] = $seq[$i];
            }
        }
        $out[] = $next;
    }
    return $out;
}

function utf8_cp($cp) {
    if ($cp < 128) return chr($cp);
    if ($cp < 2048) return chr(192 | ($cp >> 6)) . chr(128 | ($cp & 63));
    if ($cp < 65536) return chr(224 | ($cp >> 12)) . chr(128 | (($cp >> 6) & 63)) . chr(128 | ($cp & 63));
    return chr(240 | ($cp >> 18)) . chr(128 | (($cp >> 12) & 63)) . chr(128 | (($cp >> 6) & 63)) . chr(128 | ($cp & 63));
}

function byte_unicode_map() {
    $bytes = [];
    for ($i = 33; $i <= 126; $i++) $bytes[] = $i;
    for ($i = 161; $i <= 172; $i++) $bytes[] = $i;
    for ($i = 174; $i <= 255; $i++) $bytes[] = $i;
    $codes = $bytes;
    $present = array_fill_keys($bytes, true);
    $n = 0;
    for ($i = 0; $i < 256; $i++) {
        if (!isset($present[$i])) {
            $bytes[] = $i;
            $codes[] = 256 + $n;
            $n++;
        }
    }
    $map = [];
    foreach ($bytes as $i => $b) $map[$b] = utf8_cp($codes[$i]);
    return $map;
}

function build_tokenizer_files($state, $root) {
    $pieces = [];
    for ($i = 0; $i < 256; $i++) $pieces[$i] = chr($i);
    $pieces[PAD_ID] = '<pad>';
    $pieces[BOS_ID] = '<s>';
    $pieces[EOS_ID] = '</s>';
    $pieces[EN_ID] = '<|en|>';
    $pieces[FA_ID] = '<|fa|>';
    foreach ($state['merges'] as $merge) {
        [$a, $b, $n] = $merge;
        $pieces[$n] = $pieces[$a] . $pieces[$b];
    }
    $map = byte_unicode_map();
    $vocab = [];
    $specialIds = [PAD_ID, BOS_ID, EOS_ID, EN_ID, FA_ID];
    foreach ($pieces as $id => $raw) {
        if (in_array((int)$id, $specialIds, true)) continue;
        $token = '';
        for ($j = 0, $len = strlen($raw); $j < $len; $j++) $token .= $map[ord($raw[$j])];
        $pieces[$id] = $token;
        $vocab[$token] = (int)$id;
    }
    foreach ($specialIds as $id) $vocab[$pieces[$id]] = $id;
    $mergeStrings = [];
    foreach ($state['merges'] as $merge) {
        [$a, $b, $n] = $merge;
        $mergeStrings[] = $pieces[$a] . ' ' . $pieces[$b];
        $vocab[$pieces[$n]] = $n;
    }
    for ($id = (int)$state['next_id']; $id < VOCAB_TARGET; $id++) {
        $unused = '<|unused_' . $id . '|>';
        $pieces[$id] = $unused;
        $vocab[$unused] = $id;
    }
    $vocabSize = VOCAB_TARGET;
    $tokenizer = [
        'version' => '1.0',
        'truncation' => null,
        'padding' => null,
        'added_tokens' => [
            ['id' => PAD_ID, 'content' => '<pad>', 'single_word' => false, 'lstrip' => false, 'rstrip' => false, 'normalized' => false, 'special' => true],
            ['id' => BOS_ID, 'content' => '<s>', 'single_word' => false, 'lstrip' => false, 'rstrip' => false, 'normalized' => false, 'special' => true],
            ['id' => EOS_ID, 'content' => '</s>', 'single_word' => false, 'lstrip' => false, 'rstrip' => false, 'normalized' => false, 'special' => true],
            ['id' => EN_ID, 'content' => '<|en|>', 'single_word' => false, 'lstrip' => false, 'rstrip' => false, 'normalized' => false, 'special' => true],
            ['id' => FA_ID, 'content' => '<|fa|>', 'single_word' => false, 'lstrip' => false, 'rstrip' => false, 'normalized' => false, 'special' => true]
        ],
        'normalizer' => null,
        'pre_tokenizer' => ['type' => 'ByteLevel', 'add_prefix_space' => false, 'use_regex' => false],
        'post_processor' => null,
        'decoder' => ['type' => 'ByteLevel', 'add_prefix_space' => false, 'trim_offsets' => false],
        'model' => ['type' => 'BPE', 'dropout' => null, 'unk_token' => null, 'continuing_subword_prefix' => '', 'end_of_word_suffix' => '', 'fuse_unk' => false, 'vocab' => $vocab, 'merges' => $mergeStrings],
        'jibay' => ['format' => 'byte-level-bpe-v1', 'vocab_size' => $vocabSize, 'merge_ids' => $state['merges'], 'special_ids' => ['pad' => PAD_ID, 'bos' => BOS_ID, 'eos' => EOS_ID, 'en' => EN_ID, 'fa' => FA_ID]]
    ];
    atomic_write($root . '/tokenizer.json', json_encode($tokenizer, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES));
    $config = [
        'tokenizer_class' => 'PreTrainedTokenizerFast',
        'tokenizer_file' => 'tokenizer.json',
        'model_max_length' => MAX_LEN,
        'padding_side' => 'right',
        'truncation_side' => 'right',
        'clean_up_tokenization_spaces' => false,
        'add_prefix_space' => false,
        'pad_token' => '<pad>',
        'bos_token' => '<s>',
        'eos_token' => '</s>',
        'unk_token' => null,
        'additional_special_tokens' => ['<|en|>', '<|fa|>'],
        'vocab_size' => $vocabSize
    ];
    atomic_write($root . '/tokenizer_config.json', json_encode($config, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));
    atomic_write($root . '/special_tokens_map.json', json_encode(['pad_token' => '<pad>', 'bos_token' => '<s>', 'eos_token' => '</s>', 'additional_special_tokens' => ['<|en|>', '<|fa|>']], JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));
    $cfg = [
        'architectures' => [ARCH_NAME],
        'model_type' => 'jibay_for_emmbedding_lm',
        'name_or_path' => MODEL_NAME,
        'vocab_size' => $vocabSize,
        'hidden_size' => HIDDEN,
        'num_hidden_layers' => 1,
        'num_attention_heads' => HEADS,
        'head_dim' => HIDDEN / HEADS,
        'intermediate_size' => FFN,
        'hidden_act' => 'gelu',
        'max_position_embeddings' => MAX_LEN,
        'embedding_dimension' => OUT_DIM,
        'pooling_mode' => 'mean',
        'normalize_embeddings' => true,
        'position_embedding_type' => 'absolute',
        'layer_norm_eps' => 1e-5,
        'attention_dropout' => 0.0,
        'hidden_dropout' => 0.0,
        'training_objective' => 'symmetric_infonce',
        'temperature' => 0.07,
        'optimizer' => 'AdamW',
        'parameter_count' => parameter_count($vocabSize)
    ];
    atomic_write($root . '/config.json', json_encode($cfg, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));
    atomic_write($root . '/merges.txt', "#version: 0.2\n" . implode("\n", $mergeStrings) . "\n");
    atomic_write($root . '/vocab.json', json_encode($vocab, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));
}

function parameter_specs($vocab) {
    $d = HIDDEN;
    $f = FFN;
    $o = OUT_DIM;
    return [
        'E' => $vocab * $d,
        'P' => MAX_LEN * $d,
        'Wq' => $d * $d, 'bq' => $d,
        'Wk' => $d * $d, 'bk' => $d,
        'Wv' => $d * $d, 'bv' => $d,
        'Wo' => $d * $d, 'bo' => $d,
        'W1' => $d * $f, 'b1' => $f,
        'W2' => $f * $d, 'b2' => $d,
        'ln1g' => $d, 'ln1b' => $d,
        'ln2g' => $d, 'ln2b' => $d,
        'Wp' => $d * $o, 'bp' => $o
    ];
}

function parameter_count($vocab) {
    return array_sum(parameter_specs($vocab));
}

function randn_value() {
    $u1 = max(mt_rand() / mt_getrandmax(), 1e-12);
    $u2 = mt_rand() / mt_getrandmax();
    return sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
}

function initialize_model($vocab) {
    $specs = parameter_specs($vocab);
    $p = [];
    foreach ($specs as $name => $size) $p[$name] = array_fill(0, $size, 0.0);
    $d = HIDDEN;
    $f = FFN;
    $o = OUT_DIM;
    $stdEmb = 0.02;
    foreach (['E', 'P'] as $name) {
        $n = count($p[$name]);
        for ($i = 0; $i < $n; $i++) $p[$name][$i] = randn_value() * $stdEmb;
    }
    $matrices = [
        'Wq' => [$d, $d], 'Wk' => [$d, $d], 'Wv' => [$d, $d], 'Wo' => [$d, $d],
        'W1' => [$d, $f], 'W2' => [$f, $d], 'Wp' => [$d, $o]
    ];
    foreach ($matrices as $name => $shape) {
        [$in, $out] = $shape;
        $std = sqrt(2.0 / ($in + $out));
        for ($i = 0; $i < count($p[$name]); $i++) $p[$name][$i] = randn_value() * $std;
    }
    foreach (['ln1g', 'ln2g'] as $name) for ($i = 0; $i < $d; $i++) $p[$name][$i] = 1.0;
    return $p;
}

function zero_grads($params) {
    $g = [];
    foreach ($params as $name => $values) $g[$name] = array_fill(0, count($values), 0.0);
    return $g;
}

function linear_row($x, $w, $b, $in, $out) {
    $y = array_fill(0, $out, 0.0);
    for ($j = 0; $j < $out; $j++) {
        $sum = $b[$j];
        for ($i = 0; $i < $in; $i++) $sum += $x[$i] * $w[$i * $out + $j];
        $y[$j] = $sum;
    }
    return $y;
}

function layer_norm_forward($x, $gamma, $beta) {
    $n = count($x);
    $mean = array_sum($x) / $n;
    $var = 0.0;
    for ($i = 0; $i < $n; $i++) {
        $v = $x[$i] - $mean;
        $var += $v * $v;
    }
    $var /= $n;
    $inv = 1.0 / sqrt($var + 1e-5);
    $xh = [];
    $y = [];
    for ($i = 0; $i < $n; $i++) {
        $xh[$i] = ($x[$i] - $mean) * $inv;
        $y[$i] = $xh[$i] * $gamma[$i] + $beta[$i];
    }
    return [$y, $xh, $inv];
}

function layer_norm_backward($dy, $xh, $inv, $gamma, &$dgamma, &$dbeta) {
    $n = count($dy);
    $sum1 = 0.0;
    $sum2 = 0.0;
    $dxhat = [];
    for ($i = 0; $i < $n; $i++) {
        $dgamma[$i] += $dy[$i] * $xh[$i];
        $dbeta[$i] += $dy[$i];
        $dxhat[$i] = $dy[$i] * $gamma[$i];
        $sum1 += $dxhat[$i];
        $sum2 += $dxhat[$i] * $xh[$i];
    }
    $dx = [];
    for ($i = 0; $i < $n; $i++) $dx[$i] = ($inv / $n) * ($n * $dxhat[$i] - $sum1 - $xh[$i] * $sum2);
    return $dx;
}

function gelu_value($x) {
    $c = 0.7978845608028654;
    $u = $c * ($x + 0.044715 * $x * $x * $x);
    return 0.5 * $x * (1.0 + tanh($u));
}

function gelu_derivative($x) {
    $c = 0.7978845608028654;
    $u = $c * ($x + 0.044715 * $x * $x * $x);
    $t = tanh($u);
    return 0.5 * (1.0 + $t) + 0.5 * $x * (1.0 - $t * $t) * $c * (1.0 + 0.134145 * $x * $x);
}

function model_forward($tokens, $p) {
    $d = HIDDEN;
    $f = FFN;
    $o = OUT_DIM;
    $dh = intdiv($d, HEADS);
    $t = count($tokens);
    $x = $u = $q = $k = $v = $ln1xh = $ln1inv = [];
    for ($i = 0; $i < $t; $i++) {
        $id = (int)$tokens[$i];
        $row = [];
        for ($j = 0; $j < $d; $j++) $row[$j] = $p['E'][$id * $d + $j] + $p['P'][$i * $d + $j];
        $x[$i] = $row;
        [$u[$i], $ln1xh[$i], $ln1inv[$i]] = layer_norm_forward($row, $p['ln1g'], $p['ln1b']);
        $q[$i] = linear_row($u[$i], $p['Wq'], $p['bq'], $d, $d);
        $k[$i] = linear_row($u[$i], $p['Wk'], $p['bk'], $d, $d);
        $v[$i] = linear_row($u[$i], $p['Wv'], $p['bv'], $d, $d);
    }
    $probs = [];
    $ctx = array_fill(0, $t, array_fill(0, $d, 0.0));
    $scale = 1.0 / sqrt($dh);
    for ($h = 0; $h < HEADS; $h++) {
        $start = $h * $dh;
        for ($i = 0; $i < $t; $i++) {
            $scores = [];
            $maxScore = -INF;
            for ($j = 0; $j < $t; $j++) {
                $s = 0.0;
                for ($z = 0; $z < $dh; $z++) $s += $q[$i][$start + $z] * $k[$j][$start + $z];
                $scores[$j] = $s * $scale;
                if ($scores[$j] > $maxScore) $maxScore = $scores[$j];
            }
            $den = 0.0;
            for ($j = 0; $j < $t; $j++) {
                $scores[$j] = exp($scores[$j] - $maxScore);
                $den += $scores[$j];
            }
            for ($j = 0; $j < $t; $j++) $scores[$j] /= max($den, 1e-12);
            $probs[$h][$i] = $scores;
            for ($j = 0; $j < $t; $j++) {
                $pr = $scores[$j];
                for ($z = 0; $z < $dh; $z++) $ctx[$i][$start + $z] += $pr * $v[$j][$start + $z];
            }
        }
    }
    $cproj = $r = $zRows = $ln2xh = $ln2inv = $pre = $act = $ff = $hRows = [];
    for ($i = 0; $i < $t; $i++) {
        $cproj[$i] = linear_row($ctx[$i], $p['Wo'], $p['bo'], $d, $d);
        $r[$i] = [];
        for ($j = 0; $j < $d; $j++) $r[$i][$j] = $x[$i][$j] + $cproj[$i][$j];
        [$zRows[$i], $ln2xh[$i], $ln2inv[$i]] = layer_norm_forward($r[$i], $p['ln2g'], $p['ln2b']);
        $pre[$i] = linear_row($zRows[$i], $p['W1'], $p['b1'], $d, $f);
        $act[$i] = [];
        for ($j = 0; $j < $f; $j++) $act[$i][$j] = gelu_value($pre[$i][$j]);
        $ff[$i] = linear_row($act[$i], $p['W2'], $p['b2'], $f, $d);
        $hRows[$i] = [];
        for ($j = 0; $j < $d; $j++) $hRows[$i][$j] = $r[$i][$j] + $ff[$i][$j];
    }
    $pool = array_fill(0, $d, 0.0);
    for ($i = 0; $i < $t; $i++) for ($j = 0; $j < $d; $j++) $pool[$j] += $hRows[$i][$j] / $t;
    $raw = linear_row($pool, $p['Wp'], $p['bp'], $d, $o);
    $norm = sqrt(array_sum(array_map(static fn($v) => $v * $v, $raw)) + 1e-12);
    $embedding = [];
    for ($i = 0; $i < $o; $i++) $embedding[$i] = $raw[$i] / $norm;
    return [
        'embedding' => $embedding,
        'tokens' => $tokens,
        'x' => $x,
        'u' => $u,
        'q' => $q,
        'k' => $k,
        'v' => $v,
        'probs' => $probs,
        'ctx' => $ctx,
        'r' => $r,
        'z' => $zRows,
        'ln1xh' => $ln1xh,
        'ln1inv' => $ln1inv,
        'ln2xh' => $ln2xh,
        'ln2inv' => $ln2inv,
        'pre' => $pre,
        'act' => $act,
        'h' => $hRows,
        'pool' => $pool,
        'raw' => $raw,
        'norm' => $norm
    ];
}

function model_backward($cache, $gEmbedding, $p, &$g) {
    $d = HIDDEN;
    $f = FFN;
    $o = OUT_DIM;
    $dh = intdiv($d, HEADS);
    $tokens = $cache['tokens'];
    $t = count($tokens);
    $dot = 0.0;
    for ($i = 0; $i < $o; $i++) $dot += $cache['embedding'][$i] * $gEmbedding[$i];
    $gRaw = [];
    for ($i = 0; $i < $o; $i++) $gRaw[$i] = ($gEmbedding[$i] - $cache['embedding'][$i] * $dot) / $cache['norm'];
    $gPool = array_fill(0, $d, 0.0);
    for ($i = 0; $i < $d; $i++) {
        for ($j = 0; $j < $o; $j++) {
            $g['Wp'][$i * $o + $j] += $cache['pool'][$i] * $gRaw[$j];
            $gPool[$i] += $p['Wp'][$i * $o + $j] * $gRaw[$j];
        }
    }
    for ($j = 0; $j < $o; $j++) $g['bp'][$j] += $gRaw[$j];
    $dH = [];
    for ($i = 0; $i < $t; $i++) $dH[$i] = array_fill(0, $d, 0.0);
    for ($i = 0; $i < $t; $i++) for ($j = 0; $j < $d; $j++) $dH[$i][$j] = $gPool[$j] / $t;
    $dR = $dZ = [];
    for ($i = 0; $i < $t; $i++) {
        $dFF = $dH[$i];
        $dAct = array_fill(0, $f, 0.0);
        for ($a = 0; $a < $f; $a++) {
            for ($b = 0; $b < $d; $b++) {
                $g['W2'][$a * $d + $b] += $cache['act'][$i][$a] * $dFF[$b];
                $dAct[$a] += $p['W2'][$a * $d + $b] * $dFF[$b];
            }
        }
        for ($b = 0; $b < $d; $b++) $g['b2'][$b] += $dFF[$b];
        $dPre = [];
        for ($a = 0; $a < $f; $a++) $dPre[$a] = $dAct[$a] * gelu_derivative($cache['pre'][$i][$a]);
        $dZ[$i] = array_fill(0, $d, 0.0);
        for ($a = 0; $a < $d; $a++) {
            for ($b = 0; $b < $f; $b++) {
                $g['W1'][$a * $f + $b] += $cache['z'][$i][$a] * $dPre[$b];
                $dZ[$i][$a] += $p['W1'][$a * $f + $b] * $dPre[$b];
            }
        }
        for ($b = 0; $b < $f; $b++) $g['b1'][$b] += $dPre[$b];
        $dLn2 = layer_norm_backward($dZ[$i], $cache['ln2xh'][$i], $cache['ln2inv'][$i], $p['ln2g'], $g['ln2g'], $g['ln2b']);
        $dR[$i] = [];
        for ($a = 0; $a < $d; $a++) $dR[$i][$a] = $dH[$i][$a] + $dLn2[$a];
    }
    $dCtx = [];
    for ($i = 0; $i < $t; $i++) {
        $dCtx[$i] = array_fill(0, $d, 0.0);
        for ($a = 0; $a < $d; $a++) {
            for ($b = 0; $b < $d; $b++) {
                $g['Wo'][$a * $d + $b] += $cache['ctx'][$i][$a] * $dR[$i][$b];
                $dCtx[$i][$a] += $p['Wo'][$a * $d + $b] * $dR[$i][$b];
            }
            $g['bo'][$a] += $dR[$i][$a];
        }
    }
    $dQ = $dK = $dV = [];
    for ($i = 0; $i < $t; $i++) {
        $dQ[$i] = array_fill(0, $d, 0.0);
        $dK[$i] = array_fill(0, $d, 0.0);
        $dV[$i] = array_fill(0, $d, 0.0);
    }
    $scale = 1.0 / sqrt($dh);
    for ($head = 0; $head < HEADS; $head++) {
        $start = $head * $dh;
        for ($i = 0; $i < $t; $i++) {
            $dp = [];
            $weighted = 0.0;
            for ($j = 0; $j < $t; $j++) {
                $vDot = 0.0;
                for ($z = 0; $z < $dh; $z++) {
                    $vDot += $dCtx[$i][$start + $z] * $cache['v'][$j][$start + $z];
                    $dV[$j][$start + $z] += $cache['probs'][$head][$i][$j] * $dCtx[$i][$start + $z];
                }
                $dp[$j] = $vDot;
                $weighted += $cache['probs'][$head][$i][$j] * $vDot;
            }
            for ($j = 0; $j < $t; $j++) {
                $ds = $cache['probs'][$head][$i][$j] * ($dp[$j] - $weighted);
                for ($z = 0; $z < $dh; $z++) {
                    $dQ[$i][$start + $z] += $ds * $cache['k'][$j][$start + $z] * $scale;
                    $dK[$j][$start + $z] += $ds * $cache['q'][$i][$start + $z] * $scale;
                }
            }
        }
    }
    $dU = [];
    for ($i = 0; $i < $t; $i++) {
        $dU[$i] = array_fill(0, $d, 0.0);
        for ($a = 0; $a < $d; $a++) {
            for ($b = 0; $b < $d; $b++) {
                $g['Wq'][$a * $d + $b] += $cache['u'][$i][$a] * $dQ[$i][$b];
                $g['Wk'][$a * $d + $b] += $cache['u'][$i][$a] * $dK[$i][$b];
                $g['Wv'][$a * $d + $b] += $cache['u'][$i][$a] * $dV[$i][$b];
                $dU[$i][$a] += $p['Wq'][$a * $d + $b] * $dQ[$i][$b];
                $dU[$i][$a] += $p['Wk'][$a * $d + $b] * $dK[$i][$b];
                $dU[$i][$a] += $p['Wv'][$a * $d + $b] * $dV[$i][$b];
            }
            $g['bq'][$a] += $dQ[$i][$a];
            $g['bk'][$a] += $dK[$i][$a];
            $g['bv'][$a] += $dV[$i][$a];
        }
    }
    for ($i = 0; $i < $t; $i++) {
        $dLn1 = layer_norm_backward($dU[$i], $cache['ln1xh'][$i], $cache['ln1inv'][$i], $p['ln1g'], $g['ln1g'], $g['ln1b']);
        $tokenId = (int)$tokens[$i];
        for ($j = 0; $j < $d; $j++) {
            $dx = $dR[$i][$j] + $dLn1[$j];
            $g['E'][$tokenId * $d + $j] += $dx;
            $g['P'][$i * $d + $j] += $dx;
        }
    }
}

function contrastive_step($tokenBatch, &$params, &$optimizer, $step) {
    $n = count($tokenBatch);
    $caches = [];
    $emb = [];
    for ($i = 0; $i < $n; $i++) {
        $caches[$i] = model_forward($tokenBatch[$i], $params);
        $emb[$i] = $caches[$i]['embedding'];
    }
    $temperature = 0.07;
    $gradEmb = array_fill(0, $n, array_fill(0, OUT_DIM, 0.0));
    $loss = 0.0;
    for ($i = 0; $i < $n; $i++) {
        $target = ($i < intdiv($n, 2)) ? $i + intdiv($n, 2) : $i - intdiv($n, 2);
        $logits = [];
        $maxLogit = -INF;
        for ($j = 0; $j < $n; $j++) {
            $s = 0.0;
            for ($k = 0; $k < OUT_DIM; $k++) $s += $emb[$i][$k] * $emb[$j][$k];
            $logits[$j] = $s / $temperature;
            if ($logits[$j] > $maxLogit) $maxLogit = $logits[$j];
        }
        $den = 0.0;
        for ($j = 0; $j < $n; $j++) {
            $logits[$j] = exp($logits[$j] - $maxLogit);
            $den += $logits[$j];
        }
        for ($j = 0; $j < $n; $j++) $logits[$j] /= max($den, 1e-12);
        $loss -= log(max($logits[$target], 1e-12)) / $n;
        for ($j = 0; $j < $n; $j++) {
            $dlogit = ($logits[$j] - (($j === $target) ? 1.0 : 0.0)) / ($n * $temperature);
            for ($k = 0; $k < OUT_DIM; $k++) {
                $gradEmb[$i][$k] += $dlogit * $emb[$j][$k];
                $gradEmb[$j][$k] += $dlogit * $emb[$i][$k];
            }
        }
    }
    $grads = zero_grads($params);
    for ($i = 0; $i < $n; $i++) model_backward($caches[$i], $gradEmb[$i], $params, $grads);
    $norm2 = 0.0;
    foreach ($grads as $values) foreach ($values as $v) $norm2 += $v * $v;
    $gradNorm = sqrt($norm2);
    $clip = ($gradNorm > 1.0) ? 1.0 / $gradNorm : 1.0;
    $optimizer['step'] = $step;
    $lr = 0.0003 * min(1.0, $step / 100.0);
    $beta1 = 0.9;
    $beta2 = 0.999;
    $eps = 1e-8;
    foreach ($params as $name => &$values) {
        $m =& $optimizer['m'][$name];
        $v =& $optimizer['v'][$name];
        $size = count($values);
        $decay = in_array($name, ['E', 'P', 'Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2', 'Wp'], true) ? 0.01 : 0.0;
        for ($i = 0; $i < $size; $i++) {
            $gval = $grads[$name][$i] * $clip;
            $m[$i] = $beta1 * $m[$i] + (1.0 - $beta1) * $gval;
            $v[$i] = $beta2 * $v[$i] + (1.0 - $beta2) * $gval * $gval;
            $mh = $m[$i] / (1.0 - pow($beta1, $step));
            $vh = $v[$i] / (1.0 - pow($beta2, $step));
            $values[$i] -= $lr * ($mh / (sqrt($vh) + $eps) + $decay * $values[$i]);
        }
        unset($m, $v);
    }
    unset($values);
    return ['loss' => $loss, 'grad_norm' => $gradNorm, 'learning_rate' => $lr];
}

function encode_text($text, $lang, $tokenizer) {
    $ids = byte_ids($text);
    foreach ($tokenizer['jibay']['merge_ids'] as $merge) {
        [$a, $b, $newId] = $merge;
        $next = [];
        $n = count($ids);
        for ($i = 0; $i < $n; $i++) {
            if ($i + 1 < $n && $ids[$i] === $a && $ids[$i + 1] === $b) {
                $next[] = $newId;
                $i++;
            } else {
                $next[] = $ids[$i];
            }
        }
        $ids = $next;
    }
    $langId = $lang === 'fa' ? FA_ID : EN_ID;
    $available = MAX_LEN - 3;
    if (count($ids) > $available) $ids = array_slice($ids, 0, $available);
    return array_merge([BOS_ID, $langId], $ids, [EOS_ID]);
}

function detect_language($text) {
    return preg_match('/[\x{0600}-\x{06FF}\x{0750}-\x{077F}\x{08A0}-\x{08FF}]/u', $text) ? 'fa' : 'en';
}


function evaluate_pairs($pairs, $tokenizer, $params) {
    $count = count($pairs);
    if ($count < 2) return null;
    $fa = [];
    $en = [];
    foreach ($pairs as $pair) {
        $fa[] = model_forward(encode_text($pair[0], 'fa', $tokenizer), $params)['embedding'];
        $en[] = model_forward(encode_text($pair[1], 'en', $tokenizer), $params)['embedding'];
    }
    $correctFa = 0;
    $correctEn = 0;
    for ($i = 0; $i < $count; $i++) {
        $bestFa = -INF;
        $bestEn = -INF;
        $bestFaIndex = -1;
        $bestEnIndex = -1;
        for ($j = 0; $j < $count; $j++) {
            $sFa = 0.0;
            $sEn = 0.0;
            for ($k = 0; $k < OUT_DIM; $k++) {
                $sFa += $fa[$i][$k] * $en[$j][$k];
                $sEn += $en[$i][$k] * $fa[$j][$k];
            }
            if ($sFa > $bestFa) { $bestFa = $sFa; $bestFaIndex = $j; }
            if ($sEn > $bestEn) { $bestEn = $sEn; $bestEnIndex = $j; }
        }
        if ($bestFaIndex === $i) $correctFa++;
        if ($bestEnIndex === $i) $correctEn++;
    }
    return round((($correctFa + $correctEn) / (2.0 * $count)), 4);
}

function save_training_files($root, $params, $optimizer, $checkpoint) {
    save_serialized($root . '/model.weights', ['format' => 'jibay-weights-v1', 'params' => $params]);
    save_serialized($root . '/optimizer.state', $optimizer);
    atomic_write($root . '/checkpoint.json', json_encode($checkpoint, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));
    atomic_write($root . '/training_state.json', json_encode($checkpoint, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));
}

if (!is_file($DATA)) json_out(['status' => 'error', 'error' => 'dataset.txt not found'], 400);
$lockFp = fopen($LOCK, 'c');
if (!$lockFp || !flock($lockFp, LOCK_EX | LOCK_NB)) json_out(['status' => 'busy', 'message' => 'Training is already running'], 409);

$started = microtime(true);
$deadline = $started + TIME_BUDGET;
$checkpoint = is_file($CHECKPOINT) ? json_decode((string)file_get_contents($CHECKPOINT), true) : null;
if (!is_array($checkpoint)) $checkpoint = ['step' => 0, 'epoch' => 0, 'byte_offset' => 0, 'pair_index' => 0, 'trained_pairs' => 0, 'last_loss' => null, 'last_grad_norm' => null, 'status' => 'initializing'];

try {
    if (!is_file($TOKENIZER)) {
        $bpe = load_serialized($BPE_STATE);
        if ($bpe === null) {
            $bpe = read_bpe_sample($DATA, 300);
            save_serialized($BPE_STATE, $bpe);
        }
        $mergesMade = 0;
        while ($bpe['next_id'] < VOCAB_TARGET && (microtime(true) < $deadline - 1.5 || $mergesMade === 0)) {
            $counts = count_bpe_pairs($bpe['sequences']);
            if (!$counts) break;
            arsort($counts, SORT_NUMERIC);
            $key = array_key_first($counts);
            $frequency = $counts[$key];
            $left = intdiv((int)$key, 1024);
            $right = (int)$key % 1024;
            $newId = (int)$bpe['next_id'];
            $bpe['sequences'] = apply_bpe_merge($bpe['sequences'], $left, $right, $newId);
            $bpe['merges'][] = [$left, $right, $newId];
            $bpe['next_id']++;
            $mergesMade++;
            save_serialized($BPE_STATE, $bpe);
        }
        if ($bpe['next_id'] >= VOCAB_TARGET || $mergesMade === 0) {
            build_tokenizer_files($bpe, $ROOT);
            @unlink($BPE_STATE);
        } else {
            flock($lockFp, LOCK_UN);
            fclose($lockFp);
            json_out(['status' => 'tokenizer_training', 'merges_ready' => count($bpe['merges']), 'target_vocab' => VOCAB_TARGET, 'elapsed_seconds' => round(microtime(true) - $started, 2), 'message' => 'Refresh train.php to continue Byte-Level BPE training']);
        }
    }

    $tokenizer = json_decode((string)file_get_contents($TOKENIZER), true);
    if (!is_array($tokenizer) || !isset($tokenizer['jibay']['merge_ids'])) throw new RuntimeException('tokenizer.json is invalid');
    $vocab = (int)$tokenizer['jibay']['vocab_size'];
    $cfg = ['architectures' => [ARCH_NAME], 'model_type' => 'jibay_for_emmbedding_lm', 'name_or_path' => MODEL_NAME, 'vocab_size' => $vocab, 'hidden_size' => HIDDEN, 'num_hidden_layers' => 1, 'num_attention_heads' => HEADS, 'head_dim' => HIDDEN / HEADS, 'intermediate_size' => FFN, 'hidden_act' => 'gelu', 'max_position_embeddings' => MAX_LEN, 'embedding_dimension' => OUT_DIM, 'pooling_mode' => 'mean', 'normalize_embeddings' => true, 'position_embedding_type' => 'absolute', 'layer_norm_eps' => 1e-5, 'training_objective' => 'symmetric_infonce', 'temperature' => 0.07, 'optimizer' => 'AdamW', 'parameter_count' => parameter_count($vocab)];
    atomic_write($CONFIG, json_encode($cfg, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));

    $weightData = load_serialized($WEIGHTS);
    $params = ($weightData && isset($weightData['params'])) ? $weightData['params'] : initialize_model($vocab);
    $optimizer = load_serialized($OPTIMIZER);
    if ($optimizer === null || !isset($optimizer['m'], $optimizer['v'], $optimizer['step'])) {
        $optimizer = ['step' => 0, 'm' => [], 'v' => []];
        foreach ($params as $name => $values) {
            $optimizer['m'][$name] = array_fill(0, count($values), 0.0);
            $optimizer['v'][$name] = array_fill(0, count($values), 0.0);
        }
    }
    if (!isset($checkpoint['initialized']) || !$checkpoint['initialized']) {
        $checkpoint['initialized'] = true;
        $checkpoint['parameter_count'] = parameter_count($vocab);
        $checkpoint['vocab_size'] = $vocab;
        $checkpoint['architecture'] = ARCH_NAME;
        $checkpoint['model'] = MODEL_NAME;
        $checkpoint['status'] = 'training';
        save_training_files($ROOT, $params, $optimizer, $checkpoint);
    }

    $fp = fopen($DATA, 'rb');
    if (!$fp) throw new RuntimeException('Cannot open dataset.txt');
    $size = filesize($DATA);
    $offset = (int)$checkpoint['byte_offset'];
    if ($offset < 0) $offset = 0;
    if ($offset >= $size && (int)$checkpoint['epoch'] < 1) {
        $checkpoint['epoch'] = 1;
        $checkpoint['byte_offset'] = 0;
        $checkpoint['pair_index'] = 0;
    }
    if ((int)$checkpoint['epoch'] >= 1) {
        $checkpoint['status'] = 'completed';
        $checkpoint['byte_offset'] = 0;
        $checkpoint['pair_index'] = 0;
        $checkpoint['updated_at'] = date('Y-m-d H:i:s');
        $checkpoint['last_run_seconds'] = round(microtime(true) - $started, 3);
        $checkpoint['last_run_steps'] = 0;
        save_training_files($ROOT, $params, $optimizer, $checkpoint);
        fclose($fp);
        flock($lockFp, LOCK_UN);
        fclose($lockFp);
        json_out([
            'status' => 'completed',
            'model' => MODEL_NAME,
            'architecture' => ARCH_NAME,
            'parameters' => $checkpoint['parameter_count'],
            'vocab_size' => $vocab,
            'step' => $checkpoint['step'],
            'epoch' => $checkpoint['epoch'],
            'trained_pairs' => $checkpoint['trained_pairs'],
            'last_loss' => $checkpoint['last_loss'],
            'validation_recall_at_1' => $checkpoint['validation_recall_at_1'] ?? null,
            'last_gradient_norm' => $checkpoint['last_grad_norm'],
            'last_run_steps' => 0,
            'last_run_seconds' => $checkpoint['last_run_seconds'],
            'checkpoint' => 'saved',
            'message' => 'Training completed'
        ]);
    }
    fseek($fp, $offset);
    $stepsThisRun = 0;
    $validationPairs = [];
    $epochFinished = false;
    while (microtime(true) < $deadline - 1.0) {
        $batch = [];
        $nextOffset = ftell($fp);
        $newPairIndex = (int)$checkpoint['pair_index'];
        while (count($batch) < BATCH_PAIRS) {
            $pair = read_pair($fp);
            if ($pair === null) {
                $epochFinished = true;
                break;
            }
            [$fa, $en, $pairEnd] = $pair;
            $nextOffset = $pairEnd;
            $newPairIndex++;
            $splitKey = sprintf('%u', crc32($fa . "\t" . $en));
            if (((int)$splitKey % 10) === 0) {
                if (count($validationPairs) < 4) $validationPairs[] = [$fa, $en];
                continue;
            }
            $batch[] = [$fa, $en];
        }
        if (count($batch) > 0) {
            $faBatch = [];
            $enBatch = [];
            foreach ($batch as $pair) {
                $faBatch[] = encode_text($pair[0], 'fa', $tokenizer);
                $enBatch[] = encode_text($pair[1], 'en', $tokenizer);
            }
            $tokenBatch = array_merge($faBatch, $enBatch);
            $step = (int)$optimizer['step'] + 1;
            $result = contrastive_step($tokenBatch, $params, $optimizer, $step);
            $checkpoint['step'] = $step;
            $checkpoint['trained_pairs'] += count($batch);
            $checkpoint['last_loss'] = $result['loss'];
            $checkpoint['last_grad_norm'] = $result['grad_norm'];
            $checkpoint['learning_rate'] = $result['learning_rate'];
            $stepsThisRun++;
        }
        if ($epochFinished) {
            $checkpoint['epoch'] = max(1, (int)$checkpoint['epoch'] + 1);
            $checkpoint['byte_offset'] = 0;
            $checkpoint['pair_index'] = 0;
            $checkpoint['status'] = 'completed';
        } else {
            $checkpoint['byte_offset'] = $nextOffset;
            $checkpoint['pair_index'] = $newPairIndex;
            $checkpoint['status'] = 'training';
        }
        $checkpoint['updated_at'] = date('Y-m-d H:i:s');
        if (count($batch) > 0 || $epochFinished) save_training_files($ROOT, $params, $optimizer, $checkpoint);
        if ($epochFinished || count($batch) === 0) break;
    }
    fclose($fp);
    $validationRecall = evaluate_pairs($validationPairs, $tokenizer, $params);
    if ($validationRecall !== null) $checkpoint['validation_recall_at_1'] = $validationRecall;
    $checkpoint['last_run_seconds'] = round(microtime(true) - $started, 3);
    $checkpoint['last_run_steps'] = $stepsThisRun;
    if ((int)$checkpoint['epoch'] >= 1) $checkpoint['status'] = 'completed';
    else $checkpoint['status'] = 'training';
    save_training_files($ROOT, $params, $optimizer, $checkpoint);
    flock($lockFp, LOCK_UN);
    fclose($lockFp);
    json_out([
        'status' => $checkpoint['status'],
        'model' => MODEL_NAME,
        'architecture' => ARCH_NAME,
        'parameters' => $checkpoint['parameter_count'],
        'vocab_size' => $vocab,
        'step' => $checkpoint['step'],
        'epoch' => $checkpoint['epoch'],
        'trained_pairs' => $checkpoint['trained_pairs'],
        'last_loss' => $checkpoint['last_loss'],
        'validation_recall_at_1' => $checkpoint['validation_recall_at_1'] ?? null,
        'last_gradient_norm' => $checkpoint['last_grad_norm'],
        'last_run_steps' => $stepsThisRun,
        'last_run_seconds' => $checkpoint['last_run_seconds'],
        'checkpoint' => 'saved',
        'message' => $checkpoint['status'] === 'completed' ? 'Training completed' : 'Training checkpoint saved'
    ]);
} catch (Throwable $e) {
    $checkpoint['status'] = 'error';
    $checkpoint['error'] = $e->getMessage();
    $checkpoint['updated_at'] = date('Y-m-d H:i:s');
    atomic_write($CHECKPOINT, json_encode($checkpoint, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));
    flock($lockFp, LOCK_UN);
    fclose($lockFp);
    json_out(['status' => 'error', 'error' => $e->getMessage(), 'checkpoint' => 'previous saved state retained'], 500);
}
